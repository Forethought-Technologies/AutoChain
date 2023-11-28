"""Default implementation of Chain"""
import logging
from typing import Dict

from autochain.agent.structs import AgentAction, AgentFinish
from autochain.chain.base_chain import BaseChain
from autochain.errors import ToolRunningError
from autochain.tools.base import Tool
from autochain.tools.simple_handoff.tool import HandOffToAgent

logger = logging.getLogger(__name__)


class Chain(BaseChain):
    """
    Default chain with take_next_step implemented
    It handles a few common error cases with agent, such as taking repeated action with same
    inputs and whether agent should continue the conversation
    """

    return_intermediate_steps: bool = False
    handle_parsing_errors = True
    graceful_exit_tool: Tool = HandOffToAgent()

    def handle_repeated_action(self, agent_action: AgentAction) -> AgentFinish:
        print(
            f"Action taken before: {agent_action.tool}, "
            f"input: {agent_action.tool_input}"
        )
        if agent_action.model_response:
            return AgentFinish(
                message=agent_action.response,
                log=f"Action taken before: {agent_action.tool}, "
                f"input: {agent_action.tool_input}",
            )
        else:
            print("No response from agent. Gracefully exit due to repeated action")
            return AgentFinish(
                message=self.graceful_exit_tool.run(),
                log="Gracefully exit due to repeated action",
            )

    def take_next_step(
        self,
        name_to_tool_map: Dict[str, Tool],
        inputs: Dict[str, str],
    ) -> (AgentFinish, AgentAction):
        """
        How agent determines the next step after observing the inputs and intermediate steps
        Args:
            name_to_tool_map: map of tool name to the actual tool object
            inputs: a dictionary of all inputs, such as user query, past conversation and
                tools outputs

        Returns:
            Either AgentFinish to respond to user or AgentAction to take the next action
        """

        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                **inputs,
            )
        except Exception as e:
            if not self.handle_parsing_errors:
                raise e
            tool_output = f"Invalid or incomplete response due to {e}"
            print(tool_output)
            output = AgentFinish(message=self.graceful_exit_tool.run(), log=tool_output)
            return output

        if isinstance(output, AgentAction):
            output = self.agent.clarify_args_for_agent_action(output, **inputs)

        # If agent plans to respond to AgentFinish or there is a clarifying question, respond to
        # user by returning AgentFinish
        if isinstance(output, AgentFinish):
            return output

        if isinstance(output, AgentAction):
            tool_output = ""
            # Check if tool is supported
            if output.tool in name_to_tool_map:
                tool = name_to_tool_map[output.tool]

                # how to handle the case where same action with same input is taken before
                if output.tool_input == self.memory.load_memory(tool.name):
                    return self.handle_repeated_action(output)

                self.memory.save_memory(tool.name, output.tool_input)
                # We then call the tool on the tool input to get an tool_output
                try:
                    tool_output = tool.run(output.tool_input)
                except ToolRunningError as e:
                    new_agent_action = self.agent.fix_action_input(
                        tool, output, error=str(e)
                    )
                    if (
                        new_agent_action
                        and new_agent_action.tool_input != output.tool_input
                    ):
                        tool_output = tool.run(output.tool_input)

                print(
                    f"Took action '{tool.name}' with inputs '{output.tool_input}', "
                    f"and the tool_output is {tool_output}"
                )
            else:
                tool_output = f"Tool {output.tool} if not supported"

            output.tool_output = tool_output
            return output
        else:
            raise ValueError(f"Unsupported action: {type(output)}")
