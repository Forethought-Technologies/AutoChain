import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError
from uuid import UUID

import pytest
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.agent.message import AIMessage, FunctionMessage
from autochain.agent.openai_functions_agent.openai_functions_agent import (
    OpenAIFunctionsAgent,
)
from autochain.chain.chain import Chain
from autochain.memory import base as memory_module
from autochain.memory.buffer_memory import BufferMemory
from autochain.memory.redis_memory import RedisMemory
from autochain.models.base import BaseLanguageModel, Generation, LLMResult
from autochain.models.chat_openai import convert_tool_to_dict
from autochain.tools.base import Tool
from autochain.tools.internal_search.base_search_tool import BaseSearchTool
from autochain.tools.parallel_search import ENDPOINT, USER_AGENT, parallel_search_tools
from pydantic import PrivateAttr
from redis import Redis


class ScriptedModel(BaseLanguageModel):
    """Deterministic model drives real function dispatch and consumes tool results."""

    _step: int = PrivateAttr(default=0)
    first_query: str = "Python docs"

    def generate(self, messages, functions=None, stop=None):
        if len(messages) == 1 and "confidence" in messages[0].content:
            message = AIMessage(content="5")
        else:
            errors = [
                m.content
                for m in messages[-1:]
                if isinstance(m, FunctionMessage) and m.content.startswith("Parallel ")
            ]
            if errors:
                return LLMResult(
                    generations=[Generation(message=AIMessage(content=errors[-1]))]
                )
            calls = [
                ("web_search", {"query": self.first_query}),
                ("web_fetch", {"url": "https://docs.python.org/3/"}),
                ("web_search", {"query": "Python releases"}),
            ]
            if self._step < len(calls):
                name, arguments = calls[self._step]
                assert name in [tool.name for tool in functions]
                message = AIMessage(
                    content="",
                    function_call={"name": name, "arguments": json.dumps(arguments)},
                )
            else:
                outputs = [
                    m.content for m in messages if isinstance(m, FunctionMessage)
                ]
                assert any("https://docs.python.org/3/" in text for text in outputs)
                assert any("Python documentation" in text for text in outputs)
                message = AIMessage(
                    content="Python documentation: https://docs.python.org/3/"
                )
            self._step += 1
        return LLMResult(generations=[Generation(message=message)])


@contextmanager
def mcp_server(failure=None, sse=False):
    """Observe actual request bytes below Tool.run, not a mocked client helper."""
    captured = []

    def respond(request, timeout):
        assert request.full_url == ENDPOINT
        assert request.get_header("User-agent") == USER_AGENT
        assert timeout == 30.0
        assert request.get_header("Authorization") is None
        if request.get_method() == "DELETE":
            return BytesIO()
        body = json.loads(request.data)
        captured.append(body)
        method = body["method"]
        if method != "initialize":
            assert request.get_header("Mcp-session-id") == "transport-session"
            assert request.get_header("Mcp-protocol-version") == "2025-03-26"
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26"}
        elif method == "notifications/initialized":
            return BytesIO()
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            if failure:
                raise failure
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": "Python documentation https://docs.python.org/3/",
                    }
                ]
            }
        payload = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result})
        if sse:
            payload = (
                'event: message\ndata: {"jsonrpc":"2.0","method":"progress"}\n\n'
                + "data: "
                + payload
                + "\n\n"
            )

        class OpenEventStream(BytesIO):
            def readline(self, size=-1):
                line = super().readline(size)
                if not line:
                    raise TimeoutError("SSE connection remains open after the result")
                return line

        response = (
            OpenEventStream(payload.encode()) if sse else BytesIO(payload.encode())
        )
        response.headers = {
            "Mcp-Session-Id": "transport-session",
            "Content-Type": "text/event-stream" if sse else "application/json",
        }
        return response

    with mock.patch("autochain.tools.parallel_search.urlopen", side_effect=respond):
        yield captured


def make_chain(tools=None):
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=ScriptedModel(), tools=tools)
    return Chain(agent=agent, memory=BufferMemory())


def tool_calls(captured):
    return [m["params"] for m in captured if m["method"] == "tools/call"]


@pytest.mark.parametrize("factory", [ConversationalAgent, OpenAIFunctionsAgent])
def test_defaults_explicit_choices_and_function_schemas(factory, monkeypatch):
    for key in [
        "OPENAI_API_KEY",
        "PARALLEL_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CSE_ID",
    ]:
        monkeypatch.delenv(key, raising=False)
    with mock.patch("autochain.tools.parallel_search.urlopen") as network:
        agent = factory.from_llm_and_tools(llm=ScriptedModel())
        assert [t.name for t in agent.tools] == ["web_search", "web_fetch"]
        assert factory.from_llm_and_tools(llm=ScriptedModel(), tools=[]).tools == []
        chosen = Tool(
            name="my_search", description="Explicit provider", func=lambda query: query
        )
        assert [
            t.name
            for t in factory.from_llm_and_tools(
                llm=ScriptedModel(), tools=[chosen]
            ).tools
        ] == ["my_search"]
        network.assert_not_called()
    for index, tool in enumerate(agent.tools):
        argument = ["query", "url"][index]
        spec = convert_tool_to_dict(tool)
        assert spec["parameters"]["properties"] == {argument: {"type": "string"}}
        assert spec["parameters"]["required"] == [argument]


@pytest.mark.parametrize("sse", [False, True])
def test_agent_loop_consumes_results_and_preserves_conversation(sse):
    chain = make_chain()
    with mcp_server(sse=sse) as captured:
        assert "https://docs.python.org/3/" in chain.run("Research Python")["message"]
        calls = tool_calls(captured)
        assert [c["name"] for c in calls] == ["web_search", "web_fetch", "web_search"]
        assert calls[0]["arguments"]["search_queries"] == ["Python docs"]
        assert calls[1]["arguments"]["urls"] == ["https://docs.python.org/3/"]
        session = calls[0]["arguments"]["session_id"]
        assert all(c["arguments"]["session_id"] == session for c in calls)
        # A follow-up turn reuses the identity even with a fresh model instance.
        chain.agent.llm = ScriptedModel()
        chain.run("Follow up")
        assert tool_calls(captured)[-1]["arguments"]["session_id"] == session
        chain.memory.clear()
        chain.agent.llm = ScriptedModel()
        chain.run("New chat")
        assert tool_calls(captured)[-1]["arguments"]["session_id"] != session


def test_interleaved_chains_share_tools_but_not_conversation_identity():
    tools = parallel_search_tools()
    chains = [make_chain(tools), make_chain(tools)]
    with mcp_server() as captured:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda c: c.run("Research Python"), chains))
        assert all("Python documentation" in r["message"] for r in results)
        ids = [c["arguments"]["session_id"] for c in tool_calls(captured)]
        assert len(set(ids)) == 2
        assert all(ids.count(session) == 3 for session in set(ids))


@pytest.mark.parametrize(
    "failure,expected",
    [
        (HTTPError(ENDPOINT, 429, "Quota", {}, None), "HTTP 429"),
        (HTTPError(ENDPOINT, 401, "Unauthorized", {}, None), "HTTP 401"),
        (URLError("offline"), "failed or timed out"),
        (TimeoutError(), "failed or timed out"),
    ],
    ids=["quota", "authentication", "network", "timeout"],
)
def test_agent_consumes_errors_without_provider_fallback(failure, expected):
    chain = make_chain()
    with mcp_server(failure=failure) as captured:
        assert expected in chain.run("Research Python")["message"]
        assert len(tool_calls(captured)) == 1


@pytest.mark.parametrize(
    "failure",
    [TimeoutError(), HTTPError(ENDPOINT, 429, "Quota", {}, None)],
    ids=["timeout", "quota"],
)
def test_retry_retains_conversation_and_cancel_propagates(failure):
    chain = make_chain()
    with mcp_server(failure=failure) as captured:
        # A single iteration retains the failed request as an observation.
        chain.max_iterations = 1
        chain.run("Research Python")
        session = tool_calls(captured)[0]["arguments"]["session_id"]
    chain.agent.llm = ScriptedModel(first_query="Python documentation retries")
    with mcp_server() as captured:
        chain.run("Retry")
        assert tool_calls(captured)[0]["arguments"]["session_id"] == session
    chain.agent.llm = ScriptedModel(first_query="Cancel Python docs")
    with mcp_server(failure=KeyboardInterrupt()):
        with pytest.raises(KeyboardInterrupt):
            chain.run("Cancel")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "https://user:secret@example.com",
        "not a URL",
        "https://[invalid",
    ],
)
def test_invalid_urls_never_connect(url):
    with mock.patch("autochain.tools.parallel_search.urlopen") as network:
        assert "HTTP or HTTPS" in parallel_search_tools()[1].run({"url": url})
        network.assert_not_called()


class OneToolModel(BaseLanguageModel):
    action_name: str = "web_search"

    def generate(self, messages, functions=None, stop=None):
        if len(messages) == 1 and "confidence" in messages[0].content:
            message = AIMessage(content="5")
        elif isinstance(messages[-1], FunctionMessage):
            message = AIMessage(content=messages[-1].content)
        else:
            message = AIMessage(
                content="",
                function_call={
                    "name": self.action_name,
                    "arguments": '{"query": "Python docs"}',
                },
            )
        return LLMResult(generations=[Generation(message=message)])


def test_disabled_tools_do_not_dispatch():
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=OneToolModel(), tools=[])
    chain = Chain(agent=agent, memory=BufferMemory())
    with mock.patch("autochain.tools.parallel_search.urlopen") as network:
        assert "not supported" in chain.run("Search anyway")["message"]
        network.assert_not_called()


def test_explicit_provider_is_used_without_parallel_fallback():
    def configured_search(query):
        return "Configured provider's result or failure"

    selected = Tool(
        name="web_search", description="Saved provider", func=configured_search
    )
    agent = OpenAIFunctionsAgent.from_llm_and_tools(
        llm=OneToolModel(), tools=[selected]
    )
    with mock.patch("autochain.tools.parallel_search.urlopen") as network:
        result = Chain(agent=agent, memory=BufferMemory()).run("Search")
        assert result["message"] == "Configured provider's result or failure"
        network.assert_not_called()


def test_malformed_response_and_timeout_setting():
    with pytest.raises(ValueError):
        parallel_search_tools(timeout=0)
    with mock.patch("autochain.tools.parallel_search.urlopen") as network:
        response = BytesIO(b"not JSON")
        response.headers = {}
        network.return_value = response
        assert "invalid MCP response" in parallel_search_tools()[0].run("Python docs")


class ScriptedConversationModel(BaseLanguageModel):
    def generate(self, messages, functions=None, stop=None):
        prompt = messages[0].content
        if "has assistant finish helping" in prompt:
            text = "no"
        elif "has_arg_value" in prompt:
            text = json.dumps({"has_arg_value": "Yes", "clarifying_question": ""})
        elif "Python documentation https://docs.python.org/3/" in prompt:
            text = json.dumps(
                {
                    "thoughts": {"need_use_tool": "No"},
                    "response": "Python documentation https://docs.python.org/3/",
                }
            )
        else:
            text = json.dumps(
                {
                    "thoughts": {"need_use_tool": "Yes"},
                    "tool": {"name": "web_search", "args": {"query": "Python docs"}},
                }
            )
        return LLMResult(generations=[Generation(message=AIMessage(content=text))])


def test_conversational_agent_consumes_search_results():
    agent = ConversationalAgent.from_llm_and_tools(llm=ScriptedConversationModel())
    chain = Chain(agent=agent, memory=BufferMemory())
    with mcp_server() as captured:
        result = chain.run("Find Python documentation")
        assert "Python documentation https://docs.python.org/3/" in result["message"]
        assert len(tool_calls(captured)) == 1


def test_long_term_memory_never_retrieves_conversation_metadata():
    # Load the actual memory implementation without unrelated optional DB SDKs.
    backend_modules = {}
    for module_name, search_name, doc_name in [
        ("chromadb_tool", "ChromaDBSearch", "ChromaDoc"),
        ("pinecone_tool", "PineconeSearch", "PineconeDoc"),
        ("lancedb_tool", "LanceDBSeach", "LanceDBDoc"),
    ]:
        name = "autochain.tools.internal_search." + module_name
        module = types.ModuleType(name)
        setattr(module, search_name, type(search_name, (BaseSearchTool,), {}))
        setattr(module, doc_name, type(doc_name, (), {}))
        backend_modules[name] = module
    spec = importlib.util.spec_from_file_location(
        "_tested_long_term_memory",
        Path(memory_module.__file__).with_name("long_term_memory.py"),
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, backend_modules):
        spec.loader.exec_module(module)

    class Backend(BaseSearchTool):
        _queries = PrivateAttr(default_factory=list)

        def _run(self, query, top_k=1):
            return self.run({"query": query, "top_k": top_k})

        def add_docs(self, docs, **kwargs):
            pass

        def run(self, arguments):
            self._queries.append(arguments["query"])
            if arguments["query"] == "intermediate_steps":
                return []
            return "Doc 0: private customer information"

        def clear_index(self):
            pass

    memory = module.LongTermMemory(long_term_memory=Backend())
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=OneToolModel())
    chain = Chain(agent=agent, memory=memory)
    with mcp_server() as captured:
        assert "Python documentation" in chain.run("Research")["message"]
        first = tool_calls(captured)[0]["arguments"]["session_id"]
        assert str(UUID(first)) == first
        assert (
            "_autochain_parallel_conversation" not in memory.long_term_memory._queries
        )
        assert all(
            not query.startswith("_autochain")
            for query in memory.long_term_memory._queries
        )
        memory.clear()
        chain.run("New conversation")
        assert tool_calls(captured)[-1]["arguments"]["session_id"] != first


def test_redis_conversations_persist_and_reset_without_sharing_identity():
    import fnmatch

    store = {}
    client = mock.MagicMock(spec=Redis)
    client.get.side_effect = store.get

    def put(key, value, **kwargs):
        store[key] = value.encode() if isinstance(value, str) else value

    client.set.side_effect = put
    client.keys.side_effect = lambda pattern: [
        k for k in store if fnmatch.fnmatch(k, pattern)
    ]
    client.delete.side_effect = lambda key: store.pop(key, None)
    shared_tools = parallel_search_tools()
    chains = [
        Chain(
            agent=OpenAIFunctionsAgent.from_llm_and_tools(
                llm=OneToolModel(), tools=shared_tools
            ),
            memory=RedisMemory(redis_key_prefix=prefix, redis_client=client),
        )
        for prefix in ["conversation-one", "conversation-two"]
    ]
    with mcp_server() as captured:
        for chain in chains:
            chain.run("Research")
        ids = [c["arguments"]["session_id"] for c in tool_calls(captured)]
        assert len(set(ids)) == 2
        resumed = RedisMemory(redis_key_prefix="conversation-one", redis_client=client)
        assert resumed.get_conversation_id() == ids[0]
        chains[0].memory.clear()
        chains[0].run("New chat")
        assert tool_calls(captured)[-1]["arguments"]["session_id"] not in ids
