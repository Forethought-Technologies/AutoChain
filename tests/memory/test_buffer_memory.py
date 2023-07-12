from autochain.agent.message import MessageType
from autochain.memory.buffer_memory import BufferMemory


def test_buffer_kv_memory():
    memory = BufferMemory()
    memory.save_memory(key="k", value="v")
    value = memory.load_memory(key="k")
    assert value == "v"

    default_value = memory.load_memory(key="k2", default="v2")
    assert default_value == "v2"

    memory.clear()
    assert memory.load_memory(key="k") is None


def test_buffer_conversation_memory():
    memory = BufferMemory()
    memory.save_conversation("user query", MessageType.UserMessage)
    memory.save_conversation("response to user", MessageType.AIMessage)

    conversation = memory.load_conversation().format_message()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation().format_message()
    assert message_after_clear == ""
