"""
This is an example implementation of long term memory and retrieve using query
It contains three memory stores.
conversation_history stores all the messages including FunctionMessage between assistant and agent,
long_term_memory stores a collection of ChromaDoc (or would be modified use other vectory db)
kv_memory: stores anything else as kv pairs
"""
from typing import Any, Optional

from autochain.agent.message import ChatMessageHistory, MessageType
from autochain.memory.base import BaseMemory
from autochain.tools.internal_search.base_search_tool import BaseSearchTool
from autochain.tools.internal_search.chromadb_tool import ChromaDBSearch, ChromaDoc
from autochain.tools.internal_search.pinecone_tool import PineconeSearch, PineconeDoc
from autochain.tools.internal_search.lancedb_tool import LanceDBSeach, LanceDBDoc

SEARCH_PROVIDERS = (ChromaDBSearch, PineconeSearch, LanceDBSeach)
SEARCH_DOC_TYPES = (ChromaDoc, PineconeDoc, LanceDBDoc)

class LongTermMemory(BaseMemory):
    """Buffer for storing conversation memory and an in-memory kv store."""

    conversation_history = ChatMessageHistory()
    kv_memory = {}
    long_term_memory: BaseSearchTool = None

    class Config:
        keep_untouched = SEARCH_PROVIDERS

    def load_memory(
        self,
        key: Optional[str] = None,
        default: Optional[Any] = None,
        top_k: int = 1,
        **kwargs
    ) -> Any:
        """Return history buffer by key or all memories."""
        if key in self.kv_memory:
            return self.kv_memory[key]

        # else try to retrieve from long term memory
        result = self.long_term_memory.run({"query": key, "top_k": top_k})
        return result or default

    def load_conversation(self, **kwargs) -> ChatMessageHistory:
        """Return history buffer and format it into a conversational string format."""
        return self.conversation_history

    def save_memory(self, key: str, value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) > 0
            and (isinstance(value[0], SEARCH_DOC_TYPES))
        ):
            self.long_term_memory.add_docs(docs=value)
        elif key:
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
        self.long_term_memory.clear_index()
        self.kv_memory = {}
