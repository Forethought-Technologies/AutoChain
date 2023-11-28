import json
from abc import abstractmethod
from typing import Union, Any, Dict, List
from colorama import Fore

from autochain.models.base import BaseLanguageModel
from pydantic import BaseModel

from autochain.models.base import Generation
from autochain.agent.message import BaseMessage, UserMessage
from autochain.chain import constants
from autochain.utils import print_with_color


class AgentAction(BaseModel):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    """tool outputs"""
    tool_output: str = ""

    """log message for debugging"""
    log: str = ""

    """model response or """
    model_response: str = ""

    @property
    def response(self):
        """message to be stored in memory and shared with next prompt"""
        if self.model_response and not self.tool_output:
            # share the model response or log message as output if tool fails to call
            return self.model_response
        return (
            f"Outputs from using tool '{self.tool}' for inputs {self.tool_input} "
            f"is '{self.tool_output}'\n"
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

    def load_json_output(
        self,
        message: BaseMessage, 
        llm: BaseLanguageModel,
        max_retry=3
    ) -> Dict[str, Any]:
        """Try to parse JSON response from the message content."""
        text = message.content
        clean_text = self._extract_json_text(text)

        try:
            response = json.loads(clean_text)
        except Exception:
            print_with_color(
                'Generating JSON format attempt FAILED! Trying Again...', 
                Fore.RED
            )
            message = self._fix_message(clean_text)
            response = self._attempt_fix_and_generate(
                message, 
                llm, 
                max_retry, 
                attempt=0
            )

        return response
    
    @staticmethod
    def _fix_message(clean_text: str) -> UserMessage:
        '''
        If the response from model is not proper, this function should
        iteratively construct better response until response becomes json parseable
        '''
        
        # TO DO
        # Construct this message better in order to make it better iteratively by
        # _attempt_fix_and_generate recursive function
        message = UserMessage(
                    content=f"""
                        Fix the following json into correct format
                        ```json
                        {clean_text}
                        ```
                        """
                )
        return message
        
    @staticmethod
    def _extract_json_text(text: str) -> str:
        """Extract JSON text from the input string."""
        clean_text = ""
        try:
            clean_text = text[text.index("{") : text.rindex("}") + 1].strip()
        except Exception:
            clean_text = text
        return clean_text
    
    def _attempt_fix_and_generate(
        self,
        message: BaseMessage, 
        llm: BaseLanguageModel,
        max_retry: int,
        attempt: int
    ) -> Dict[str, Any]:
        
        """Attempt to fix JSON format using model generation recursively."""
        if attempt >= max_retry:
            raise ValueError(
                """
                Max retry reached. Model is unable to generate proper JSON output. 
                Try with another Model!
                """
            )

        full_output: Generation = llm.generate([message]).generations[0]

        try:
            response = json.loads(full_output.message.content)
            return response
        except Exception:
            print_with_color(
                'Generating JSON format attempt FAILED! Trying Again...',
                Fore.RED
            )
            clean_text = self._extract_json_text(full_output.message.content)
            message = self._fix_message(clean_text)
            return self._attempt_fix_and_generate(message, llm, max_retry, attempt=attempt + 1)

    @abstractmethod
    def parse(self, message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""

    def parse_clarification(
        self, message: BaseMessage, agent_action: AgentAction
    ) -> Union[AgentAction, AgentFinish]:
        """Parse clarification outputs"""
        return agent_action
