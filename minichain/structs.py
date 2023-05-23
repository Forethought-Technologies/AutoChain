from typing import Union, Any, Dict, List

from pydantic import BaseModel


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
        return f"Observation from using tool '{self.tool}' for inputs {self.tool_input} " \
               f"is '{self.observation}'\n"


class AgentFinish(BaseModel):
    """Agent's return value."""

    return_values: dict
    log: str
    intermediate_steps: List[AgentAction] = []

    def format_output(self) -> Dict[str, Any]:
        final_output = self.return_values
        final_output["intermediate_steps"] = self.intermediate_steps
        return final_output
