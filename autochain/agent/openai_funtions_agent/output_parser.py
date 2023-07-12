import json
from typing import Union

from autochain.agent.message import AIMessage
from autochain.agent.structs import AgentAction, AgentFinish, AgentOutputParser


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
