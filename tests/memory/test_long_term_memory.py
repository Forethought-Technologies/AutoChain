from autochain.agent.message import MessageType
from autochain.memory.long_term_memory import LongTermMemory
from autochain.tools.internal_search.chromadb_tool import ChromaDoc, ChromaDBSearch
from autochain.tools.internal_search.pinecone_tool import PineconeSearch, PineconeDoc
from autochain.tools.internal_search.lancedb_tool import LanceDBSeach, LanceDBDoc
from test_utils.pinecone_mocks import DummyEncoder, pinecone_index_fixture


def test_long_term_kv_memory_chromadb():
    memory = LongTermMemory(
        long_term_memory=ChromaDBSearch(docs=[], description="long term memory")
    )
    memory.save_memory(key="k", value="v")
    value = memory.load_memory(key="k")
    assert value == "v"

    default_value = memory.load_memory(key="k2", default="v2")
    assert default_value == "v2"

    memory.clear()
    assert memory.load_memory(key="k") is None


def test_buffer_conversation_memory():
    memory = LongTermMemory(
        long_term_memory=ChromaDBSearch(docs=[], description="long term memory")
    )
    memory.save_conversation("user query", MessageType.UserMessage)
    memory.save_conversation("response to user", MessageType.AIMessage)

    conversation = memory.load_conversation().format_message()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation().format_message()
    assert message_after_clear == ""


def test_long_term_memory():
    d = ChromaDoc("This is document1", metadata={"source": "notion"})
    memory = LongTermMemory(
        long_term_memory=ChromaDBSearch(docs=[], description="long term memory")
    )
    memory.save_memory(key="", value=[d])

    value = memory.load_memory(key="document query")
    assert value == "Doc 0: This is document1"


def test_long_term_kv_memory_pincode(pinecone_index_fixture):
    memory = LongTermMemory(
        long_term_memory=PineconeSearch(
            docs=[], description="long term memory", encoder=DummyEncoder()
        )
    )
    memory.save_memory(key="k", value="v")
    value = memory.load_memory(key="k")
    assert value == "v"

    default_value = memory.load_memory(key="k2", default="v2")
    assert default_value == "v2"

    memory.clear()
    assert memory.load_memory(key="k") is None


def test_buffer_conversation_memory_pinecone(pinecone_index_fixture):
    memory = LongTermMemory(
        long_term_memory=PineconeSearch(
            docs=[], description="long term memory", encoder=DummyEncoder()
        )
    )
    memory.save_conversation("user query", MessageType.UserMessage)
    memory.save_conversation("response to user", MessageType.AIMessage)

    conversation = memory.load_conversation().format_message()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation().format_message()
    assert message_after_clear == ""


def test_long_term_memory_pinecone(pinecone_index_fixture):
    d = PineconeDoc(
        "This is document1",
    )
    memory = LongTermMemory(
        long_term_memory=PineconeSearch(
            docs=[], description="long term memory", encoder=DummyEncoder()
        )
    )
    memory.save_memory(key="", value=[d])

    value = memory.load_memory(key="document query")
    assert value == "Doc 0: This is document1"

def test_long_term_kv_memory_lancedb():
    memory = LongTermMemory(
        long_term_memory=LanceDBSeach(
            docs=[], description="long term memory", encoder=DummyEncoder()
        )
    )
    memory.save_memory(key="k", value="v")
    value = memory.load_memory(key="k")
    assert value == "v"

    default_value = memory.load_memory(key="k2", default="v2")
    assert default_value == "v2"

    memory.clear()
    assert memory.load_memory(key="k") is None


def test_buffer_conversation_memory_lancedb():
    memory = LongTermMemory(
        long_term_memory=LanceDBSeach(
            docs=[], description="long term memory", encoder=DummyEncoder()
        )
    )
    memory.save_conversation("user query", MessageType.UserMessage)
    memory.save_conversation("response to user", MessageType.AIMessage)

    conversation = memory.load_conversation().format_message()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation().format_message()
    assert message_after_clear == ""


def test_long_term_memory_lancedb():
    d = LanceDBDoc(
        "This is document1",
    )
    memory = LongTermMemory(
        long_term_memory=LanceDBSeach(
            docs=[], description="long term memory", encoder=DummyEncoder()
        )
    )
    memory.save_memory(key="", value=[d])

    value = memory.load_memory(key="document query")
    assert value == "Doc 0: This is document1"
