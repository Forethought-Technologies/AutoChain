import json
import logging
import re
from typing import Union

from autochain.agent.message import AIMessage
from autochain.agent.structs import AgentAction, AgentFinish, AgentOutputParser

logger = logging.getLogger(__name__)


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

    def parse_estimated_confidence(self, message: AIMessage) -> int:
        """Parse estimated confidence from the message"""

        def find_first_integer(input_string):
            # Define a regular expression pattern to match integers
            pattern = re.compile(r"\d+")

            # Search for the first match in the input string
            match = pattern.search(input_string)

            # Check if a match is found
            if match:
                # Extract and return the matched integer
                return int(match.group())
            else:
                # Return 0 if no integer is found
                logger.info(f"\nCannot find confidence in message: {input_string}\n")
                return 0

        content = message.content.strip()

        return find_first_integer(content)
