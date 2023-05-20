"""Common schema objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
)

from pydantic import BaseModel


class BaseMemory(BaseModel, ABC):
    """Base interface for memory in chains."""

    @abstractmethod
    def load_memory(self, key: str, **kwargs) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """

    @abstractmethod
    def load_conversation(self, **kwargs) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """

    @abstractmethod
    def save_memory(self, key: str, value: Any) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def save_conversation(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""
