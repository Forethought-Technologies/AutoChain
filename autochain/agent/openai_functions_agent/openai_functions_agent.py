from __future__ import annotations

import logging
from string import Template
from typing import Any, Dict, List, Optional, Union

from autochain.agent.base_agent import BaseAgent
from autochain.agent.message import ChatMessageHistory, SystemMessage, UserMessage
from autochain.agent.openai_functions_agent.output_parser import (
    OpenAIFunctionOutputParser,
)
from autochain.agent.openai_functions_agent.prompt import ESTIMATE_CONFIDENCE_PROMPT
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.models.base import BaseLanguageModel, Generation
from autochain.tools.base import Tool
from autochain.utils import print_with_color
from colorama import Fore

logger = logging.getLogger(__name__)


class OpenAIFunctionsAgent(BaseAgent):
    """
    Agent supports function calling natively in OpenAI, which leverage function message to
    determine which tool should be used
    When tool is not selected, responds just like conversational agent
    Tool descriptions are generated from typing from the tool
    """

    llm: BaseLanguageModel = None
    allowed_tools: Dict[str, Tool] = {}
    tools: List[Tool] = []
    prompt: Optional[str] = None
    min_confidence: int = 3

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[List[Tool]] = None,
        output_parser: Optional[OpenAIFunctionOutputParser] = None,
        prompt: str = None,
        min_confidence: int = 3,
        **kwargs: Any,
    ) -> OpenAIFunctionsAgent:
        tools = tools or []

        allowed_tools = {tool.name: tool for tool in tools}
        _output_parser = output_parser or OpenAIFunctionOutputParser()
        return cls(
            llm=llm,
            allowed_tools=allowed_tools,
            output_parser=_output_parser,
            tools=tools,
            prompt=prompt,
            min_confidence=min_confidence,
            **kwargs,
        )

    def plan(
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        retries: int = 2,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        while retries > 0:
            print_with_color("Planning", Fore.LIGHTYELLOW_EX)

            final_messages = []
            if self.prompt:
                final_messages.append(SystemMessage(content=self.prompt))
            final_messages += history.messages

            logger.info(f"\nPlanning Input: {[m.content for m in final_messages]} \n")
            full_output: Generation = self.llm.generate(
                final_messages, self.tools
            ).generations[0]

            agent_output: Union[AgentAction, AgentFinish] = self.output_parser.parse(
                full_output.message
            )
            print(
                f"Planning output: \nmessage content: {repr(full_output.message.content)}; "
                f"function_call: "
                f"{repr(full_output.message.function_call)}",
                Fore.YELLOW,
            )
            if isinstance(agent_output, AgentAction):
                print_with_color(
                    f"Plan to take action '{agent_output.tool}'", Fore.LIGHTYELLOW_EX
                )

            generation_is_confident = self.is_generation_confident(
                history=history,
                agent_output=agent_output,
                min_confidence=self.min_confidence,
            )
            if not generation_is_confident:
                retries -= 1
                print_with_color(
                    f"Generation is not confident, {retries} retries left",
                    Fore.LIGHTYELLOW_EX,
                )
                continue
            else:
                return agent_output

    def is_generation_confident(
        self,
        history: ChatMessageHistory,
        agent_output: Union[AgentAction, AgentFinish],
        min_confidence: int = 3,
    ) -> bool:
        """
        Estimate the confidence of the generation
        Args:
            history: history of the conversation
            agent_output: the output from the agent
            min_confidence: minimum confidence score to be considered as confident
        """

        def _format_assistant_message(action_output: Union[AgentAction, AgentFinish]):
            if isinstance(action_output, AgentFinish):
                assistant_message = f"Assistant: {action_output.message}"
            elif isinstance(action_output, AgentAction):
                assistant_message = f"Action: {action_output.tool} with input: {action_output.tool_input}"
            else:
                raise ValueError("Unsupported action for estimating confidence score")

            return assistant_message

        prompt = Template(ESTIMATE_CONFIDENCE_PROMPT).substitute(
            policy=self.prompt,
            conversation_history=history.format_message(),
            assistant_message=_format_assistant_message(agent_output),
        )
        logger.info(f"\nEstimate confidence prompt: {prompt} \n")

        message = UserMessage(content=prompt)

        full_output: Generation = self.llm.generate([message], self.tools).generations[
            0
        ]

        estimated_confidence = self.output_parser.parse_estimated_confidence(
            full_output.message
        )

        return estimated_confidence >= min_confidence
