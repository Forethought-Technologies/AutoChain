import json
from abc import abstractmethod
from typing import Union

from pydantic import Extra

from minichain.agent.prompt import SBS_FORMAT_INSTRUCTIONS
from minichain.errors import OutputParserException
from minichain.structs import AgentAction, AgentFinish
from minichain.tools.tools import HandOffToAgent


class AgentOutputParser():
    @abstractmethod
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""


class ConvoJSONOutputParser(AgentOutputParser):

    def get_format_instructions(self) -> str:
        return SBS_FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            clean_text = text[text.index("{"):text.rindex("}") + 1].strip()
            response = json.loads(clean_text)
            # print(f"Response: {response}\n")
        except Exception as e:
            raise OutputParserException(f"Not a valid json: `{text}`")

        handoff_action = HandOffToAgent()
        if ("no" in response.get("thoughts", {}).get("need_use_tool").lower().strip()
            or "yes" not in response.get("validation", {}).get("arg_valid").lower()
        ):
            output_message = response.get("response")
            if output_message:
                return AgentFinish(return_values={
                    "output": response.get("response"),
                }, log=output_message)
            else:
                return AgentAction(tool=handoff_action.name, tool_input={}, log="Empty model "
                                                                                "response",
                                   response=output_message)

        action_name = response.get("tool", {}).get("name")
        action_args = response.get("tool", {}).get("args")
        return AgentAction(tool=action_name, tool_input=action_args,
                           log=f"Previous plan: {response.get('thoughts', {}).get('plan')}\n"
                           f"Previous tool used: {response.get('tool', {}).get('name')}\n",
                           response=response.get("response"))
