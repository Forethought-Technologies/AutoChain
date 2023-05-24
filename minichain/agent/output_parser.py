import json
from abc import abstractmethod
from typing import Union

from pydantic import BaseModel

from minichain.agent.prompt import SBS_INSTRUCTION_FORMAT
from minichain.errors import OutputParserException
from minichain.structs import AgentAction, AgentFinish
from minichain.tools.tools import HandOffToAgent


class AgentOutputParser(BaseModel):
    instruction_format: str = ""

    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""


class ConvoJSONOutputParser(AgentOutputParser):
    instruction_format = SBS_INSTRUCTION_FORMAT

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            clean_text = text[text.index("{"):text.rindex("}") + 1].strip()
            response = json.loads(clean_text)
        except Exception:
            raise OutputParserException(f"Not a valid json: `{text}`")

        handoff_action = HandOffToAgent()
        action_name = response.get("tool", {}).get("name")
        action_args = response.get("tool", {}).get("args")
        if action_name == handoff_action.name:
            return AgentAction(tool=handoff_action.name, tool_input={},
                               log="Needs to hand off",
                               model_response=response.get("response", ""))

        if ("no" in response.get("thoughts", {}).get("need_use_tool").lower().strip()
            or "yes" not in response.get("validation", {}).get("arg_valid").lower()
        ):
            output_message = response.get("response")
            if output_message:
                return AgentFinish(return_values={
                    "output": response.get("response"),
                }, log=output_message)
            else:
                return AgentAction(tool=handoff_action.name,
                                   tool_input={}, log="Empty model response",
                                   model_response=output_message)

        return AgentAction(tool=action_name,
                           tool_input=action_args,
                           model_response=response.get("response", ""))

    def parse_clarification(self, text: str,
                            agent_action: AgentAction) -> Union[AgentAction, AgentFinish]:
        try:
            clean_text = text[text.index("{"):text.rindex("}") + 1].strip()
            response = json.loads(clean_text)
        except Exception:
            raise OutputParserException(f"Not a valid json: `{text}`")

        need_clarification = response.get('need_clarification', "")
        clarifying_question = response.get('clarifying_question', "")

        if "yes" in need_clarification.lower() and clarifying_question:
            return AgentFinish(return_values={
                "output": clarifying_question,
            }, log=clarifying_question)
        else:
            return agent_action
