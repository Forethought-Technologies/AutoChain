"""Common memory schema object."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Optional,
    Union,
)
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from autochain.agent.message import ChatMessageHistory, MessageType


class BaseMemory(BaseModel, ABC):
    """Base interface for memory in chains."""

    conversation_id: UUID = Field(default_factory=uuid4)
    """Opaque conversation metadata, separate from retrievable user memories."""

    def get_conversation_id(self) -> str:
        """Return conversation identity without semantic retrieval."""
        return str(self.conversation_id)

    def reset_conversation_id(self) -> None:
        """Start a fresh identity. Custom memory implementations call this in clear()."""
        self.conversation_id = uuid4()

    @abstractmethod
    def load_memory(
        self, key: Union[str, None] = None, default: Optional[Any] = None, **kwargs: Any
    ) -> Any:
        """Return key-value pairs given the text input to the chain."""

    @abstractmethod
    def load_conversation(self, **kwargs) -> ChatMessageHistory:
        """Return key-value pairs given the text input to the chain."""

    @abstractmethod
    def save_memory(self, key: str, value: Any) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def save_conversation(
        self, message: str, message_type: MessageType, **kwargs
    ) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""
