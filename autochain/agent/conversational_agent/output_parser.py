import json
from typing import Union

from colorama import Fore

from autochain.agent.message import BaseMessage
from autochain.agent.structs import AgentAction, AgentFinish, AgentOutputParser
from autochain.errors import OutputParserException
from autochain.utils import print_with_color


class ConvoJSONOutputParser(AgentOutputParser):
    def parse(self, message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        response = self.load_json_output(message)

        action_name = response.get("tool", {}).get("name")
        action_args = response.get("tool", {}).get("args")

        if (
            "no" in response.get("thoughts", {}).get("need_use_tool").lower().strip()
            or not action_name
        ):
            output_message = response.get("response")
            if output_message:
                return AgentFinish(message=response.get("response"), log=output_message)
            else:
                return AgentFinish(
                    message="Sorry, i don't understand", log=output_message
                )

        return AgentAction(
            tool=action_name,
            tool_input=action_args,
            model_response=response.get("response", ""),
        )

    def parse_clarification(
        self, message: BaseMessage, agent_action: AgentAction
    ) -> Union[AgentAction, AgentFinish]:
        response = self.load_json_output(message)

        has_arg_value = response.get("has_arg_value", "")
        clarifying_question = response.get("clarifying_question", "")

        if "no" in has_arg_value.lower() and clarifying_question:
            return AgentFinish(message=clarifying_question, log=clarifying_question)
        else:
            return agent_action
