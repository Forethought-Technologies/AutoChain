from collections import defaultdict
from typing import Any, Dict, Optional

from minichain.agent.message import ChatMessageHistory
from minichain.memory.base import BaseMemory


class BufferMemory(BaseMemory):
    """Buffer for storing conversation memory and an in-memory kv store."""

    chat_history = ChatMessageHistory()
    entire_history = defaultdict(list)
    kv_memory = {}

    def load_memory(
        self, key: Optional[str] = None, default: Optional[Any] = None, **kwargs
    ) -> Any:
        """Return history buffer by key or all memories."""
        if not key:
            return self.kv_memory

        return self.kv_memory.get(key, default)

    def load_conversation(self, **kwargs) -> Any:
        """Return history buffer and format it into a conversational string format."""
        return self.chat_history.format_message()

    def save_memory(self, key: str, value: Any) -> None:
        self.kv_memory[key] = value

    def save_conversation(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer."""
        self.entire_history["input"].append(inputs)
        self.entire_history["output"].append(outputs)
        self.chat_history.add_user_message(inputs["query"])
        self.chat_history.add_ai_message(outputs["message"])

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_history.clear()
        self.entire_history = defaultdict(list)
        self.kv_memory = {}
