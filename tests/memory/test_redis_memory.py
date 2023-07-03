from typing import Any, List

from autochain.agent.message import MessageType, UserMessage, AIMessage
from autochain.memory.redis_memory import RedisMemory
from unittest.mock import patch, MagicMock, Mock

from redis.client import Redis

import pickle


class MockRedis:
    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Any:
        if key in self.cache:
            return self.cache[key]
        return None

    def set(self, key: str, value: Any, ex: int) -> Any:
        if self.cache:
            self.cache[key] = value
            return "OK"
        return None

    def keys(self, key: str) -> List[str]:
        idx = key.index("*")
        if idx is not None:
            key = key[:idx]
            return [k for k in self.cache.keys() if k.startswith(key)]
        else:
            return [k for k in self.cache.keys() if k == key]

    def delete(self, key) -> None:
        del self.cache[key]


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
    mock_redis.get.side_effect = [None, None, pickle.dumps([user_message, ai_message]), None]

    memory = RedisMemory(redis_key_prefix="test", redis_client=mock_redis)
    memory.save_conversation(user_query, MessageType.UserMessage)
    memory.save_conversation(ai_response, MessageType.AIMessage)

    conversation = memory.load_conversation().format_message()
    assert conversation == "User: user query\nAssistant: response to user\n"

    memory.clear()
    message_after_clear = memory.load_conversation().format_message()
    assert message_after_clear == ""
