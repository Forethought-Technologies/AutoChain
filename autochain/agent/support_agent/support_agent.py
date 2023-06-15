from __future__ import annotations

import json
import logging
from string import Template
from typing import Any, Dict, List, Optional, Union

from colorama import Fore
from autochain.agent.base_agent import BaseAgent
from autochain.agent.message import BaseMessage, UserMessage, ChatMessageHistory
from autochain.agent.prompt_formatter import JSONPromptTemplate
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.agent.support_agent.output_parser import SupportJSONOutputParser
from autochain.agent.support_agent.prompt import (
    CLARIFYING_QUESTION_PROMPT,
    FIX_TOOL_INPUT_PROMPT_FORMAT,
    PLANNING_PROMPT,
    SHOULD_ANSWER_PROMPT,
)
from autochain.models.base import BaseLanguageModel, Generation
from autochain.tools.base import Tool
from autochain.tools.simple_handoff.tool import HandOffToAgent
from autochain.utils import print_with_color

logger = logging.getLogger(__name__)


class SupportAgent(BaseAgent):
    """
    SupportAgent is a type of agent that tries to answer user question using a list of tools
    The main difference with conversational agent is different prompt and handling when agent
    is not sure how to answer the question. It has a special variable in prompt called "policy",
    which determines the main logic agent should follow
    """

    output_parser: SupportJSONOutputParser = SupportJSONOutputParser()
    llm: BaseLanguageModel = None
    prompt_template: JSONPromptTemplate = None
    allowed_tools: Dict[str, Tool] = {}
    tools: List[Tool] = []

    # injected policy agent should follow into the prompt
    policy: str = ""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[List[Tool]] = None,
        output_parser: Optional[SupportJSONOutputParser] = None,
        prompt: str = PLANNING_PROMPT,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> SupportAgent:
        """Construct an agent from an LLM and tools."""
        tools = tools or []
        tools.append(HandOffToAgent())

        prompt_template = cls.get_prompt_template(
            prompt=prompt,
            input_variables=input_variables,
        )

        allowed_tools = {tool.name: tool for tool in tools}
        _output_parser = output_parser or SupportJSONOutputParser()
        return cls(
            llm=llm,
            allowed_tools=allowed_tools,
            output_parser=_output_parser,
            prompt_template=prompt_template,
            tools=tools,
            **kwargs,
        )

    def should_answer(
        self, should_answer_prompt_template: str = SHOULD_ANSWER_PROMPT, **kwargs
    ) -> Optional[AgentFinish]:
        """Determine if agent should continue to answer user questions based on the latest user
        query"""
        if "query" not in kwargs or "history" not in kwargs or not kwargs["history"]:
            return None

        def _parse_response(res: str):
            if "yes" in res.lower():
                return AgentFinish(
                    message="Thank your for contacting",
                    log="Thank your for contacting",
                )
            else:
                return None

        prompt = Template(should_answer_prompt_template).substitute(**kwargs)
        response = (
            self.llm.generate([UserMessage(content=prompt)])
            .generations[0]
            .message.content
        )
        return _parse_response(response)

    @staticmethod
    def format_prompt(
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
            input_variables = ["input", "agent_scratchpad"]
        return JSONPromptTemplate(template=template, input_variables=input_variables)

    def plan(
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Plan the next step. either taking an action with AgentAction or respond to user with AgentFinish
        Args:
            history:
            intermediate_steps: List of AgentAction that has been performed with outputs
            **kwargs: key value pairs from chain, which contains query and other stored memories

        Returns:
            AgentAction or AgentFinish
        """
        print_with_color("Planning", Fore.LIGHTYELLOW_EX)
        tool_names = ", ".join([tool.name for tool in self.tools])
        tool_strings = "\n\n".join(
            [f"> {tool.name}: \n{tool.description}" for tool in self.tools]
        )
        inputs = {
            "tool_names": tool_names,
            "tools": tool_strings,
            "policy": self.policy,
            "history": history.format_message(),
            **kwargs,
        }
        final_messages = self.format_prompt(
            self.prompt_template, intermediate_steps, **inputs
        )
        logger.info(f"\nFull Input: {[m.content for m in final_messages]} \n")

        full_output: Generation = self.llm.generate(final_messages).generations[0]
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

            # call hand off to agent and finish workflow
            if agent_output.tool == HandOffToAgent().name:
                return AgentFinish(
                    message=HandOffToAgent().run(), log="Handing off to agent"
                )

        return agent_output

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
        print_with_color("Deciding if need clarification", Fore.LIGHTYELLOW_EX)
        inputs = {
            "tool_name": agent_action.tool,
            "tool_desp": self.allowed_tools.get(agent_action.tool).description,
            "history": history.format_message(),
            **kwargs,
        }

        clarifying_template = self.get_prompt_template(
            prompt=CLARIFYING_QUESTION_PROMPT
        )

        final_prompt = self.format_prompt(
            clarifying_template, intermediate_steps, **inputs
        )
        logger.info(f"\nClarification inputs: {final_prompt[0].content}")
        full_output: Generation = self.llm.generate(final_prompt).generations[0]
        return self.output_parser.parse_clarification(
            full_output.message, agent_action=agent_action
        )

    def fix_action_input(
        self, tool: Tool, action: AgentAction, error: str
    ) -> AgentAction:
        prompt = FIX_TOOL_INPUT_PROMPT_FORMAT.format(
            tool_description=tool.description, inputs=action.tool_input, error=error
        )

        logger.info(f"\nFixing tool input prompt: {prompt}")
        messages = UserMessage(content=prompt)
        output = self.llm.generate([messages])
        text = output.generations[0].message.content
        inputs = text[text.index("{") : text.rindex("}") + 1].strip()
        new_tool_inputs = json.loads(inputs)

        logger.info(f"\nFixed tool output: {new_tool_inputs}")
        new_action = AgentAction(tool=action.tool, tool_input=new_tool_inputs)
        return new_action
