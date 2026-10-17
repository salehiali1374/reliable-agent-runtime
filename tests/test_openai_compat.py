import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_runtime import Message, ToolCall
from agent_runtime.llm import LLMError, TransientLLMError
from agent_runtime.openai_compat import OpenAICompatClient

from .conftest import read_file


@pytest.fixture(autouse=True)
def _bypass_proxy(monkeypatch):
    """The fake server is on localhost; a proxy configured in the environment must not see it."""
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")


class FakeServer:
    """Serves queued (status, body) responses and records the requests it got."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                status, body = outer.responses.pop(0)
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(responses):
        server = FakeServer(responses)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def chat(message, usage=None):
    return 200, {"choices": [{"message": message}], "usage": usage or {}}


def test_text_answer_and_usage_are_parsed(serve):
    server = serve([chat({"content": "done"}, {"prompt_tokens": 7, "completion_tokens": 3})])
    client = OpenAICompatClient(server.url, "m", api_key="secret")
    turn = client.complete([Message(role="user", content="hi")], [])

    assert turn.content == "done"
    assert turn.tool_calls == []
    assert (turn.usage.prompt_tokens, turn.usage.completion_tokens) == (7, 3)
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer secret"
    assert "tools" not in request["body"]


def test_tool_call_is_parsed_and_specs_are_sent(serve):
    reply = {
        "content": None,
        "tool_calls": [
            {
                "id": "abc",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "app.py"}'},
            }
        ],
    }
    server = serve([chat(reply)])
    turn = OpenAICompatClient(server.url, "m").complete(
        [Message(role="user", content="x")], [read_file.spec]
    )

    assert turn.tool_calls == [ToolCall(id="abc", name="read_file", arguments={"path": "app.py"})]
    sent = server.requests[0]["body"]["tools"][0]["function"]
    assert sent["name"] == "read_file"
    assert sent["parameters"]["required"] == ["path"]


def test_malformed_argument_json_is_flagged_not_fatal(serve):
    reply = {"tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": "{oops"}}]}
    turn = OpenAICompatClient(serve([chat(reply)]).url, "m").complete([], [])
    assert turn.tool_calls[0].parse_error


def test_history_is_converted_to_wire_format(serve):
    server = serve([chat({"content": "ok"})])
    history = [
        Message(role="system", content="s"),
        Message(role="user", content="u"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a"})],
        ),
        Message(role="tool", content="text", tool_call_id="c1"),
    ]
    OpenAICompatClient(server.url, "m").complete(history, [])

    wire = server.requests[0]["body"]["messages"]
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"path": "a"}'
    assert wire[2]["content"] is None
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "text"}


@pytest.mark.parametrize("status", [429, 500, 503])
def test_server_overload_is_transient(serve, status):
    client = OpenAICompatClient(serve([(status, {"error": "busy"})]).url, "m")
    with pytest.raises(TransientLLMError):
        client.complete([], [])


@pytest.mark.parametrize("status", [400, 401, 404])
def test_client_errors_are_permanent(serve, status):
    client = OpenAICompatClient(serve([(status, {"error": "no"})]).url, "m")
    with pytest.raises(LLMError) as info:
        client.complete([], [])
    assert not isinstance(info.value, TransientLLMError)


def test_connection_failure_is_transient():
    client = OpenAICompatClient("http://127.0.0.1:1/v1", "m", timeout=1)
    with pytest.raises(TransientLLMError):
        client.complete([], [])


def test_unexpected_response_shape_is_permanent(serve):
    client = OpenAICompatClient(serve([(200, {"nope": 1})]).url, "m")
    with pytest.raises(LLMError, match="unexpected response"):
        client.complete([], [])
