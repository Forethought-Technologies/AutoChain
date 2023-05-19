from collections import defaultdict
from typing import Any, Dict

from minichain.memory.base import BaseMemory
from minichain.memory.message import ChatMessageHistory


class BufferMemory(BaseMemory):
    """Buffer for storing conversation memory."""
    chat_history = ChatMessageHistory()
    entire_history = defaultdict(list)
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    def load_memory(self, **kwargs) -> Dict[str, Any]:
        """Return history buffer."""
        history = self.chat_history.format_message()

        return {'history': history}

    def save_memory(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""

        self.entire_history['input'].append(inputs)
        self.entire_history['output'].append(outputs)
        self.chat_history.add_user_message(inputs['query'])
        self.chat_history.add_ai_message(outputs['output'])

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_history.clear()
        self.entire_history = {}
