import enum
from abc import abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MessageType(enum.Enum):
    UserMessage = enum.auto()
    AIMessage = enum.auto()
    SystemMessage = enum.auto()
    FunctionMessage = enum.auto()


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
    function_call: Dict[str, Any] = {}

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


class FunctionMessage(BaseMessage):
    """Type of message that is a function message."""

    name: str
    conversational_message: str = ""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "function"


class ChatMessageHistory(BaseModel):
    messages: List[BaseMessage] = []

    def save_message(self, message: str, message_type: MessageType, **kwargs):
        if message_type == MessageType.AIMessage:
            self.messages.append(AIMessage(content=message))
        elif message_type == MessageType.UserMessage:
            self.messages.append(UserMessage(content=message))
        elif message_type == MessageType.FunctionMessage:
            self.messages.append(
                FunctionMessage(
                    content=message,
                    name=kwargs["name"],
                    conversational_message=kwargs["conversational_message"],
                )
            )
        elif message_type == MessageType.SystemMessage:
            self.messages.append(SystemMessage(content=message))

    def format_message(self):
        string_messages = []
        if len(self.messages) > 0:
            for m in self.messages:
                if isinstance(m, FunctionMessage):
                    string_messages.append(f"Action: {m.conversational_message}")
                    continue

                if isinstance(m, UserMessage):
                    role = "User"
                elif isinstance(m, AIMessage):
                    role = "Assistant"
                elif isinstance(m, SystemMessage):
                    role = "System"
                else:
                    continue
                string_messages.append(f"{role}: {m.content}")
            return "\n".join(string_messages) + "\n"
        return ""

    def get_latest_user_message(self) -> UserMessage:
        for message in reversed(self.messages):
            if isinstance(message, UserMessage):
                return message
        return UserMessage(content="n/a")

    def clear(self) -> None:
        self.messages = []
