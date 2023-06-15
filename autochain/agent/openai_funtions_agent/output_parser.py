import json
from typing import Union

from colorama import Fore

from autochain.agent.message import BaseMessage, AIMessage
from autochain.agent.structs import AgentAction, AgentFinish, AgentOutputParser
from autochain.errors import OutputParserException
from autochain.utils import print_with_color


class OpenAIFunctionOutputParser(AgentOutputParser):
    def parse(self, message: AIMessage) -> Union[AgentAction, AgentFinish]:
        if message.function_call:
            action_name = message.function_call["name"]
            action_args = json.loads(message.function_call["arguments"])

            return AgentAction(
                tool=action_name,
                tool_input=action_args,
                model_response=message.content,
            )
        else:
            return AgentFinish(message=message.content, log=message.content)

    @staticmethod
    def parse_clarification(
        message: BaseMessage, agent_action: AgentAction
    ) -> Union[AgentAction, AgentFinish]:
        text = message.content
        try:
            clean_text = text[text.index("{") : text.rindex("}") + 1].strip()
            response = json.loads(clean_text)
            print_with_color(f"Full clarification output: {response}", Fore.YELLOW)
        except Exception:
            raise OutputParserException(f"Not a valid json: `{text}`")

        has_arg_value = response.get("has_arg_value", "")
        clarifying_question = response.get("clarifying_question", "")

        if "no" in has_arg_value.lower() and clarifying_question:
            return AgentFinish(message=clarifying_question, log=clarifying_question)
        else:
            return agent_action
