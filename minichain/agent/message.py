from abc import abstractmethod
from typing import List

from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""


class UserMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "user"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class ChatMessageHistory(BaseModel):
    messages: List[BaseMessage] = []

    def add_user_message(self, message: str) -> None:
        self.messages.append(UserMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def format_message(self):
        string_messages = []
        for m in self.messages:
            if isinstance(m, UserMessage):
                role = "User"
            elif isinstance(m, AIMessage):
                role = "Assistant"
            elif isinstance(m, SystemMessage):
                role = "System"
            else:
                raise ValueError(f"Got unsupported message type: {m}")
            string_messages.append(f"{role}: {m.content}")
        return "\n".join(string_messages)

    def clear(self) -> None:
        self.messages = []
