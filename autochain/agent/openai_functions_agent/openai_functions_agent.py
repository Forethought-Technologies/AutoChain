from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from colorama import Fore

from autochain.agent.base_agent import BaseAgent
from autochain.agent.message import ChatMessageHistory, SystemMessage
from autochain.agent.openai_functions_agent.output_parser import (
    OpenAIFunctionOutputParser,
)
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.models.base import BaseLanguageModel, Generation
from autochain.tools.base import Tool
from autochain.utils import print_with_color

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

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[List[Tool]] = None,
        output_parser: Optional[OpenAIFunctionOutputParser] = None,
        prompt: str = None,
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
            **kwargs,
        )

    def plan(
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
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

        return agent_output
