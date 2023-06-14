from __future__ import annotations

import json
import logging
from string import Template
from typing import Any, Dict, List, Optional, Union

from colorama import Fore

from autochain.agent.base_agent import BaseAgent
from autochain.agent.message import BaseMessage
from autochain.agent.openai_funtion_agent.output_parser import (
    OpenAIFunctionOutputParser,
)
from autochain.agent.prompt_formatter import JSONPromptTemplate
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.agent.openai_funtion_agent.prompt import (
    PLANNING_PROMPT,
)
from autochain.models.base import BaseLanguageModel, Generation
from autochain.tools.base import Tool
from autochain.utils import print_with_color

logger = logging.getLogger(__name__)


class OpenAIFunctionAgent(BaseAgent):
    """ """

    llm: BaseLanguageModel = None
    prompt_template: JSONPromptTemplate = None
    allowed_tools: Dict[str, Tool] = {}
    tools: List[Tool] = []

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[List[Tool]] = None,
        output_parser: Optional[OpenAIFunctionOutputParser] = None,
        prompt: str = PLANNING_PROMPT,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> OpenAIFunctionAgent:
        tools = tools or []

        prompt_template = cls.get_prompt_template(
            prompt=prompt,
            input_variables=input_variables,
        )

        allowed_tools = {tool.name: tool for tool in tools}
        _output_parser = output_parser or OpenAIFunctionOutputParser()
        return cls(
            llm=llm,
            allowed_tools=allowed_tools,
            output_parser=_output_parser,
            prompt_template=prompt_template,
            tools=tools,
            **kwargs,
        )

    @staticmethod
    def get_final_prompt(
        template: JSONPromptTemplate,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> List[BaseMessage]:
        def _construct_scratchpad(
            actions: List[AgentAction],
        ) -> Union[str, List[BaseMessage]]:
            scratchpad = ""
            for action in actions:
                scratchpad += action.response
            return scratchpad

        """Create the full inputs for the LLMChain from intermediate steps."""
        thoughts = _construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts}
        full_inputs = {**kwargs, **new_inputs}
        prompt = template.format_prompt(**full_inputs)
        return prompt

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
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return JSONPromptTemplate(template=template, input_variables=input_variables)

    def plan(
        self, intermediate_steps: List[AgentAction], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        print_with_color("Planning", Fore.LIGHTYELLOW_EX)
        final_prompt = self.get_final_prompt(
            self.prompt_template, intermediate_steps, **kwargs
        )

        logger.info(f"\nFull Input: {final_prompt[0].content} \n")
        full_output: Generation = self.llm.generate(
            final_prompt, self.tools
        ).generations[0]
        agent_output: Union[AgentAction, AgentFinish] = self.output_parser.parse(
            full_output.message
        )

        print_with_color(
            f"Full output: {json.loads(full_output.message.content)}", Fore.YELLOW
        )
        if isinstance(agent_output, AgentAction):
            print_with_color(
                f"Plan to take action '{agent_output.tool}'", Fore.LIGHTYELLOW_EX
            )

        return agent_output
