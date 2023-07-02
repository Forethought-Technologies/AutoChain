from typing import Any, Optional

from autochain.agent.message import ChatMessageHistory, MessageType
from autochain.memory.base import BaseMemory


class BufferMemory(BaseMemory):
    """Buffer for storing conversation memory and an in-memory kv store."""

    conversation_history = ChatMessageHistory()
    kv_memory = {}

    def load_memory(
        self, key: Optional[str] = None, default: Optional[Any] = None, **kwargs
    ) -> Any:
        """Return history buffer by key or all memories."""
        if not key:
            return self.kv_memory

        return self.kv_memory.get(key, default)

    def load_conversation(self, **kwargs) -> ChatMessageHistory:
        """Return history buffer and format it into a conversational string format."""
        return self.conversation_history

    def save_memory(self, key: str, value: Any) -> None:
        self.kv_memory[key] = value

    def save_conversation(
        self, message: str, message_type: MessageType, **kwargs
    ) -> None:
        """Save context from this conversation to buffer."""
        self.conversation_history.save_message(
            message=message, message_type=message_type, **kwargs
        )

    def clear(self) -> None:
        """Clear memory contents."""
        self.conversation_history.clear()
        self.kv_memory = {}
