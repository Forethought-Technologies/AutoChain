import json
from abc import abstractmethod
from typing import Union, Any, Dict, List

from autochain.errors import OutputParserException
from pydantic import BaseModel

from autochain.agent.message import BaseMessage
from autochain.chain import constants
from autochain.models.base import Generation


class AgentAction(BaseModel):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    """tool outputs"""
    observation: str = ""

    """log message for debugging"""
    log: str = ""

    """model response or """
    model_response: str = ""

    @property
    def response(self):
        """message to be stored in memory and shared with next prompt"""
        if self.model_response and not self.observation:
            # share the model response or log message as output if tool fails to call
            return self.model_response
        return (
            f"Observation from using tool '{self.tool}' for inputs {self.tool_input} "
            f"is '{self.observation}'\n"
        )


class AgentFinish(BaseModel):
    """Agent's return value."""

    message: str
    log: str
    intermediate_steps: List[AgentAction] = []

    def format_output(self) -> Dict[str, Any]:
        final_output = {
            "message": self.message,
            constants.INTERMEDIATE_STEPS: self.intermediate_steps,
        }
        return final_output


class AgentOutputParser(BaseModel):
    @staticmethod
    def load_json_output(message: BaseMessage) -> Dict[str, Any]:
        """If the message contains a json response, try to parse it into dictionary"""
        text = message.content
        try:
            clean_text = text[text.index("{") : text.rindex("}") + 1].strip()
            response = json.loads(clean_text)
        except Exception:
            raise OutputParserException(f"Not a valid json: `{text}`")

        return response

    @abstractmethod
    def parse(self, message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""

    def parse_clarification(
        self, message: BaseMessage, agent_action: AgentAction
    ) -> Union[AgentAction, AgentFinish]:
        """Parse clarification outputs"""
        return agent_action
