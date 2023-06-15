"""Common memory schema object."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Optional,
    Union,
)

from pydantic import BaseModel

from autochain.agent.message import ChatMessageHistory, MessageType


class BaseMemory(BaseModel, ABC):
    """Base interface for memory in chains."""

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
