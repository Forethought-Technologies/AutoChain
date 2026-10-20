"""Keyless search and page extraction through Parallel's HTTP MCP endpoint."""
import json
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from autochain.tools.base import Tool

ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "AutoChain (https://github.com/Forethought-Technologies/AutoChain)"
_conversation = ContextVar("autochain_parallel_conversation", default=None)
_MAX_RESPONSE = 2 * 1024 * 1024


@contextmanager
def search_conversation(memory):
    """Bind search identity to chain memory; clearing memory starts a new chat."""
    session_id = memory.get_conversation_id() if memory is not None else str(uuid4())
    token = _conversation.set(session_id)
    try:
        yield
    finally:
        _conversation.reset(token)


class _Client:
    """Small synchronous Streamable HTTP client for this fixed server only."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.session = None

    def request(self, method, params, request_id=None):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
            headers["MCP-Protocol-Version"] = "2025-03-26"
        request = Request(ENDPOINT, data=json.dumps(message).encode(), headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            if method == "initialize":
                self.session = response.headers.get("Mcp-Session-Id")
            if request_id is None:
                return None
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                # Stop at the matching event even if the server keeps SSE open.
                size, data = 0, []
                result = None
                while size <= _MAX_RESPONSE:
                    line = response.readline(_MAX_RESPONSE - size + 1)
                    size += len(line)
                    if size > _MAX_RESPONSE:
                        raise ValueError("Parallel response exceeds 2 MiB")
                    if not line:
                        break
                    text = line.decode().rstrip("\r\n")
                    if text.startswith("data:"):
                        data.append(text[5:].lstrip(" "))
                    elif not text and data:
                        message = json.loads("\n".join(data))
                        data = []
                        if (
                            isinstance(message, dict)
                            and message.get("id") == request_id
                        ):
                            result = message
                            break
            else:
                body = response.read(_MAX_RESPONSE + 1)
                if len(body) > _MAX_RESPONSE:
                    raise ValueError("Parallel response exceeds 2 MiB")
                result = json.loads(body)
            if (
                not isinstance(result, dict)
                or result.get("id") != request_id
                or result.get("jsonrpc") != "2.0"
            ):
                raise ValueError("Invalid Parallel MCP response")
            if "error" in result:
                raise ValueError("Parallel MCP returned an error")
            if not isinstance(result.get("result"), dict):
                raise ValueError("Invalid Parallel MCP result")
            return result["result"]

    def call(self, name, arguments):
        try:
            initialized = self.request(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "autochain", "version": "0.0.5"},
                },
                1,
            )
            if initialized.get("protocolVersion") != "2025-03-26":
                raise ValueError("Unsupported Parallel MCP protocol version")
            self.request("notifications/initialized", {})
            tools = self.request("tools/list", {}, 2)["tools"]
            if name not in [tool["name"] for tool in tools]:
                raise ValueError("Parallel MCP tool is unavailable")
            # Conversation identity is separate from the MCP transport session.
            arguments["session_id"] = _conversation.get() or str(uuid4())
            result = self.request(
                "tools/call", {"name": name, "arguments": arguments}, 3
            )
            text = "\n".join(
                item["text"]
                for item in result.get("content", [])
                if item["type"] == "text"
            )
            if result.get("isError"):
                return "Parallel tool error: " + text
            if not text:
                raise ValueError("Parallel returned no text")
            return text
        except HTTPError as error:
            return f"Parallel unavailable (HTTP {error.code}); retry later or use another configured tool."
        except (URLError, TimeoutError, socket.timeout):
            return "Parallel request failed or timed out; retry later."
        except (ValueError, KeyError, TypeError):
            return "Parallel returned an invalid MCP response; retry later."
        finally:
            if self.session:
                request = Request(
                    ENDPOINT,
                    method="DELETE",
                    headers={
                        "Mcp-Session-Id": self.session,
                        "MCP-Protocol-Version": "2025-03-26",
                        "User-Agent": USER_AGENT,
                    },
                )
                try:
                    with urlopen(request, timeout=self.timeout):
                        pass
                except (OSError, ValueError):
                    pass  # Session cleanup must not hide the tool's result.


def parallel_search_tools(timeout: float = 30.0) -> List[Tool]:
    """Create search/fetch tools without connecting or reading ambient API keys.

    The timeout is in seconds per HTTP request. Standalone calls each start a
    research run; calls dispatched by a Chain share its conversation identity.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    # Plain functions keep compatibility with ChatOpenAI's signature inspection.
    def web_search(query: str):
        if not query.strip():
            return "Provide a nonempty search query."
        return _Client(timeout).call(
            "web_search", {"objective": query, "search_queries": [query]}
        )

    def web_fetch(url: str):
        try:
            parsed = urlsplit(url)
        except ValueError:
            return "Provide an HTTP or HTTPS URL without credentials."
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return "Provide an HTTP or HTTPS URL without credentials."
        return _Client(timeout).call("web_fetch", {"urls": [url]})

    return [
        Tool(
            name="web_search",
            description="Search the web for current information. Input: a search query. Returns source URLs and excerpts. Powered by Parallel Search MCP.",
            func=web_search,
        ),
        Tool(
            name="web_fetch",
            description="Read a web page for evidence. Input: an HTTP or HTTPS URL, usually from web_search. Returns page excerpts.",
            func=web_fetch,
        ),
    ]
