from typing import Union, Any, Dict, List

from pydantic import BaseModel


class AgentAction(BaseModel):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str
    response: str = ""
    observation: str = ""


class AgentFinish(BaseModel):
    """Agent's return value."""

    return_values: dict
    log: str
    intermediate_steps: List[AgentAction] = []

    def format_output(self) -> Dict[str, Any]:
        final_output = self.return_values
        final_output["intermediate_steps"] = self.intermediate_steps
        return final_output
