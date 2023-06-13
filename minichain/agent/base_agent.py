from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from pydantic import BaseModel, Extra

from minichain.agent.structs import AgentAction, AgentFinish, AgentOutputParser
from minichain.models.base import BaseLanguageModel
from minichain.tools.base import Tool


class BaseAgent(BaseModel, ABC):
    output_parser: AgentOutputParser = None
    llm: BaseLanguageModel = None
    tools: Sequence[Tool] = []

    class Config:
        """Configuration for this pydantic object."""

    extra = Extra.forbid
    arbitrary_types_allowed = True

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[Tool],
        prompt: str,
        output_parser: Optional[AgentOutputParser] = None,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseAgent:
        """Construct an agent from an LLM and tools."""

    def should_answer(
        self, should_answer_prompt_template: str = "", **kwargs
    ) -> Optional[AgentFinish]:
        """Determine if agent should continue to answer user questions based on the latest user
        query"""
        return None

    @abstractmethod
    def plan(
        self, intermediate_steps: List[AgentAction], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Plan for the next step"""

    @abstractmethod
    def clarify_args_for_agent_action(
        self,
        agent_action: AgentAction,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Ask clarifying question if needed. When agent is about to perform an action, we could
        use this function with different prompt to ask clarifying question for input if needed.
        Sometimes the planning response would already have the clarifying question, but we found
        it is more precise if there is a different prompt just for clarifying args

        Args:
            agent_action: agent action about to take
            intermediate_steps: observations so far
            **kwargs:

        Returns:
            Either a clarifying question (AgentFinish) or take the planned action (AgentAction)
        """

    def fix_action_input(
        self, tool: Tool, action: AgentAction, error: str
    ) -> Optional[AgentAction]:
        """If the tool failed due to error, what should be the fix for inputs"""
        pass
