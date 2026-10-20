import pickle
from typing import Any, Optional, Dict
from uuid import UUID

from autochain.agent.message import (
    ChatMessageHistory,
    MessageType,
    BaseMessage,
    AIMessage,
    UserMessage,
    FunctionMessage,
    SystemMessage,
)
from autochain.memory.base import BaseMemory
from redis import Redis

from autochain.memory.constants import ONE_HOUR


class RedisMemory(BaseMemory):
    """Store conversation info in redis memory."""

    expire_time: int = ONE_HOUR
    redis_key_prefix: str
    redis_client: Redis

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def load_memory(
        self, key: Optional[str] = None, default: Optional[Any] = None, **kwargs
    ) -> Any:
        """Get the key's corresponding value from redis."""
        if not key.startswith(self.redis_key_prefix):
            key = self.redis_key_prefix + f":{key}"
        pickled = self.redis_client.get(key)
        if not pickled:
            return default
        return pickle.loads(pickled)

    def load_conversation(self, **kwargs: Dict[str, Any]) -> ChatMessageHistory:
        """Return chat message history."""
        redis_key = self.redis_key_prefix + f":{ChatMessageHistory.__name__}"
        return ChatMessageHistory(messages=self.load_memory(redis_key, []))

    def save_memory(self, key: str, value: Any) -> None:
        """Save the key value pair to redis."""
        if not key.startswith(self.redis_key_prefix):
            key = self.redis_key_prefix + f":{key}"
        pickled = pickle.dumps(value)
        self.redis_client.set(key, pickled, ex=self.expire_time)

    def get_conversation_id(self) -> str:
        """Persist typed metadata independently of pickled conversation contents."""
        key = self.redis_key_prefix + ":_autochain_conversation_id"
        saved = self.redis_client.get(key)
        try:
            if isinstance(saved, bytes):
                saved = saved.decode("ascii")
            if not isinstance(saved, str):
                raise ValueError("Missing conversation identity")
            self.conversation_id = UUID(saved)
        except (ValueError, UnicodeError):
            pass
        self.redis_client.set(key, str(self.conversation_id), ex=self.expire_time)
        return str(self.conversation_id)

    def save_conversation(
        self, message: str, message_type: MessageType, **kwargs
    ) -> None:
        """Save context from this conversation to redis."""
        redis_key = self.redis_key_prefix + f":{ChatMessageHistory.__name__}"
        pickled = self.redis_client.get(redis_key)
        if pickled:
            messages: list[BaseMessage] = pickle.loads(pickled)
        else:
            messages = []
        if message_type == MessageType.AIMessage:
            messages.append(AIMessage(content=message))
        elif message_type == MessageType.UserMessage:
            messages.append(UserMessage(content=message))
        elif message_type == MessageType.FunctionMessage:
            messages.append(FunctionMessage(content=message, name=kwargs["name"]))
        elif message_type == MessageType.SystemMessage:
            messages.append(SystemMessage(content=message))
        else:
            raise ValueError(f"Unsupported message type: {message_type}")
        self.save_memory(redis_key, messages)

    def clear(self) -> None:
        """Clear redis memory."""
        for key in self.redis_client.keys(f"{self.redis_key_prefix}:*"):
            self.redis_client.delete(key)
        self.reset_conversation_id()
