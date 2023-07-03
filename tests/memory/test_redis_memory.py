import pickle
from unittest.mock import MagicMock

from autochain.agent.message import AIMessage, MessageType, UserMessage
from autochain.memory.redis_memory import RedisMemory
from redis.client import Redis


def test_redis_kv_memory():
    mock_redis = MagicMock(spec=Redis)
    pickled = pickle.dumps("v")
    mock_redis.get.side_effect = [pickled, None, None]

    memory = RedisMemory(redis_key_prefix="test", redis_client=mock_redis)

    memory.save_memory(key="k", value="v")
    value = memory.load_memory(key="k")
    assert value == "v"

    default_value = memory.load_memory(key="k2", default="v2")
    assert default_value == "v2"

    memory.clear()
    assert memory.load_memory(key="k") is None


def test_redis_conversation_memory():
    mock_redis = MagicMock(spec=Redis)
    user_query = "user query"
    ai_response = "response to user"
    user_message = UserMessage(content=user_query)
    ai_message = AIMessage(content=ai_response)
    mock_redis.get.side_effect = [
        None,
        None,
        pickle.dumps([user_message, ai_message]),
        None,
    ]

    memory = RedisMemory(redis_key_prefix="test", redis_client=mock_redis)
    memory.save_conversation(user_query, MessageType.UserMessage)
    memory.save_conversation(ai_response, MessageType.AIMessage)

    conversation = memory.load_conversation().format_message()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation().format_message()
    assert message_after_clear == ""
