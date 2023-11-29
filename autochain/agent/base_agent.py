from __future__ import annotations

from abc import ABC, abstractmethod
from string import Template
from typing import Any, List, Optional, Sequence, Union

from autochain.agent.message import ChatMessageHistory
from autochain.agent.prompt_formatter import JSONPromptTemplate
from autochain.agent.structs import AgentAction, AgentFinish, AgentOutputParser
from autochain.models.base import BaseLanguageModel
from autochain.tools.base import Tool
from pydantic import BaseModel


class BaseAgent(BaseModel, ABC):
    output_parser: AgentOutputParser = None
    llm: BaseLanguageModel = None
    tools: Sequence[Tool] = []

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
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Plan the next step. either taking an action with AgentAction or respond to user with AgentFinish
        Args:
            history: entire conversation history between user and agent including the latest query
            intermediate_steps: List of AgentAction that has been performed with outputs
            **kwargs: key value pairs from chain, which contains query and other stored memories

        Returns:
            AgentAction or AgentFinish
        """

    def clarify_args_for_agent_action(
        self,
        agent_action: AgentAction,
        history: ChatMessageHistory,
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
            history: conversation history including the latest query
            intermediate_steps: list of agent action taken so far
            **kwargs:

        Returns:
            Either a clarifying question (AgentFinish) or take the planned action (AgentAction)
        """
        return agent_action

    def fix_action_input(
        self, tool: Tool, action: AgentAction, error: str
    ) -> Optional[AgentAction]:
        """If the tool failed due to error, what should be the fix for inputs"""
        pass

    @staticmethod
    def get_prompt_template(
        prompt: str = "",
        input_variables: Optional[List[str]] = None,
    ) -> JSONPromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            prompt: message to be injected between prefix and suffix.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        template = Template(prompt)

        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return JSONPromptTemplate(template=template, input_variables=input_variables)

    def is_generation_confident(
        self,
        history: ChatMessageHistory,
        agent_output: Union[AgentAction, AgentFinish],
        min_confidence: int = 3,
    ) -> bool:
        """Check if the generation is confident enough to take action"""
        return True
