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
    def load_memory(self, **kwargs) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """

    @abstractmethod
    def save_memory(self, **kwargs) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""
