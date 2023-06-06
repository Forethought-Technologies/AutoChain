"""Base interface that all chains should implement."""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from pydantic import BaseModel

from minichain.agent.conversational_agent.conversational_agent import ConversationalAgent
from minichain.agent.structs import AgentAction, AgentFinish
from minichain.chain import constants
from minichain.errors import ToolRunningError
from minichain.memory.base import BaseMemory
from minichain.tools.base import Tool
from minichain.tools.simple_handoff.tools import HandOffToAgent


class BaseChain(BaseModel, ABC):
    """
    Base interface that all chains should implement.
    Chain will standardize inputs and outputs, the main entry pointy is the run function.
    """

    agent: Optional[ConversationalAgent] = None
    tools: List[Tool] = []
    memory: Optional[BaseMemory] = None
    verbosity: str = ""
    last_query: str = ""
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None

    def prep_inputs(self, user_query: str) -> Dict[str, str]:
        """Load conversation history from memory and prep inputs."""
        inputs = {
            "query": user_query,
        }
        if self.memory is not None:
            conversation_history = self.memory.load_conversation()
            inputs.update({constants.CONVERSATION_HISTORY: conversation_history})
        return inputs

    def prep_output(
        self,
        inputs: Dict[str, str],
        output: AgentFinish,
        return_only_outputs: bool = False,
    ) -> Dict[str, Any]:
        """Save conversation into memory and prep outputs."""
        output_dict = output.format_output()
        if self.memory is not None:
            self.memory.save_conversation(inputs=inputs, outputs=output_dict)
            self.memory.save_memory(key=constants.OBSERVATIONS, value=output.intermediate_steps)

        if return_only_outputs:
            return output_dict
        else:
            return {**inputs, **output_dict}

    def run(
        self,
        user_query: str,
        return_only_outputs: bool = False,
    ) -> Dict[str, Any]:
        """Wrapper for _run function by formatting the input and outputs

        Args:
            user_query: user query
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = self.prep_inputs(user_query)

        try:
            output = self._run(inputs)
        except (KeyboardInterrupt, Exception) as e:
            raise e

        return self.prep_output(inputs, output, return_only_outputs)

    def _run(
        self,
        inputs: Dict[str, Any],
    ) -> AgentFinish:
        """
        Run inputs including user query and past conversation with agent and get response back
        calls _take_next_step function to determine what should be the next step after
        collecting all the inputs and memorized contents
        """
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}

        intermediate_steps: List[AgentAction] = self.memory.load_memory(constants.OBSERVATIONS, [])
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            print(f"\nInputs: {inputs}\n Intermediate steps: {intermediate_steps}\n")
            next_step_output = self._should_answer(inputs=inputs)

            # if next_step_output is None which means should ask agent to answer and take next
            # step
            if not next_step_output:
                next_step_output = self._take_next_step(
                    name_to_tool_map,
                    inputs,
                    intermediate_steps,
                )

            if isinstance(next_step_output, AgentFinish):
                next_step_output.intermediate_steps = intermediate_steps
                return next_step_output

            intermediate_steps.append(next_step_output)
            iterations += 1
            time_elapsed = time.time() - start_time

        # force the termination when shouldn't continue
        output = AgentFinish(
            message="Agent stopped due to iteration limit or time limit.",
            log="",
            intermediate_steps=intermediate_steps
        )
        return output

    @abstractmethod
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, Tool],
        inputs: Dict[str, str],
        intermediate_steps: List[AgentAction],
    ) -> (AgentFinish, AgentAction):
        """How agent determines the next step after observing the inputs and intermediate
        steps"""

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _should_answer(self, inputs) -> Optional[AgentFinish]:
        """
        Let agent determines if it should continue to answer questions
        or that is the end of the conversation
        Args:
            inputs: Dict contains user query and other memorized contents

        Returns:
            None if should answer
            AgentFinish if should NOT answer and respond to user with message
        """
        output = None
        # check if agent should answer this query
        if self.last_query != inputs['query']:
            output = self.agent.should_answer(**inputs)
            self.last_query = inputs['query']

        return output


class Chain(BaseChain):
    """
    Default chain with _take_next_step implemented
    It handles a few common error cases with agent, such as taking repeated action with same
    inputs and whether agent should continue the conversation
    """

    return_intermediate_steps: bool = False
    handle_parsing_errors = True

    @staticmethod
    def handle_repeated_action(agent_action: AgentAction) -> AgentFinish:
        if agent_action.model_response:
            print(f"Action taken before: {agent_action.tool}, "
                  f"input: {agent_action.tool_input}")
            return AgentFinish(
                message=agent_action.response,
                log=f"Action taken before: {agent_action.tool}, "
                    f"input: {agent_action.tool_input}"
            )
        else:
            return AgentFinish(
                message=HandOffToAgent().run(""),
                log=f"Handing off to agent"
            )

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, Tool],
        inputs: Dict[str, str],
        intermediate_steps: List[AgentAction],
    ) -> (AgentFinish, AgentAction):
        """
        How agent determines the next step after observing the inputs and intermediate steps
        Args:
            name_to_tool_map: map of tool name to the actual tool object
            inputs: a dictionary of all inputs, such as user query, past conversation and
                observations
            intermediate_steps: list of actions and observations previously have taken

        Returns:
            Either AgentFinish to respond to user or AgentAction to take the next action
        """

        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                **inputs,
            )
        except Exception as e:
            if not self.handle_parsing_errors:
                raise e
            observation = f"Invalid or incomplete response due to {e}"
            print(observation)
            output = AgentFinish(message=HandOffToAgent().run(""), log=observation)
            return output

        if isinstance(output, AgentAction):
            output = self.agent.clarify_args_for_agent_action(output,
                                                              intermediate_steps,
                                                              **inputs)

        # If agent plans to respond to AgentFinish or there is a clarifying question, respond to
        # user by returning AgentFinish
        if isinstance(output, AgentFinish):
            return output

        if isinstance(output, AgentAction):
            observation = ""
            # Check if tool is supported
            if output.tool in name_to_tool_map:
                tool = name_to_tool_map[output.tool]

                # how to handle the case where same action with same input is taken before
                if output.tool_input == self.memory.load_memory(tool.name):
                    return self.handle_repeated_action(output)

                self.memory.save_memory(tool.name, output.tool_input)
                # We then call the tool on the tool input to get an observation
                try:
                    observation = tool.run(output.tool_input)
                except ToolRunningError as e:
                    new_agent_action = self.agent.fix_action_input(tool, output,
                                                                   error=str(e))
                    if new_agent_action and new_agent_action.tool_input != output.tool_input:
                        observation = tool.run(output.tool_input)

            else:
                observation = f"Tool {output.tool} if not supported"

            output.observation = observation
            return output
        else:
            raise ValueError(f"Unsupported action: {type(output)}")
