from autochain.chain import constants
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
    memory.save_conversation(
        inputs={
            constants.CONVERSATION_HISTORY: "conversation history",
            "query": "user query",
        },
        outputs={
            "message": "response to user",
            constants.INTERMEDIATE_STEPS: [],
        },
    )

    conversation = memory.load_conversation()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation()
    assert message_after_clear == ""
