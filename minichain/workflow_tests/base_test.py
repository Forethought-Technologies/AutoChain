import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import pandas as pd
from colorama import Fore

from minichain.agent.conversational_agent import ConversationalAgent
from minichain.agent.message import UserMessage
from minichain.chain import constants
from minichain.chain.chain import Chain
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.base import Generation
from minichain.models.chat_openai import ChatOpenAI
from minichain.tools.base import Tool
from minichain.utils import print_with_color


@dataclass
class TestCase:
    test_name: str = ""
    user_query: str = ""
    user_context: str = ""
    expected_outcome: str = ""


class BaseTest(ABC):
    @property
    @abstractmethod
    def policy(self) -> str:
        """Workflow policy"""

    @property
    @abstractmethod
    def tools(self) -> List[Tool]:
        """Workflow policy"""

    @property
    def agent_cls(self):
        """Specify agent used for this workflow"""
        return ConversationalAgent

    @property
    @abstractmethod
    def test_cases(self):
        """"""


class WorkflowTester:

    def __init__(self, tests: List[BaseTest], output_dir: str):
        self.agent_chain = None
        self.tests = tests
        self.output_dir = output_dir
        self.memory = BufferMemory()
        self.llm = ChatOpenAI(temperature=0)

    def test_each_case(self, test_case: TestCase):
        self.memory.clear()

        user_query = test_case.user_query
        conversation_history = [("user", user_query)]
        print_with_color(f">> User: {test_case.user_query}", Fore.GREEN)

        conversation_end = False
        max_turn = 8
        response = {}
        while not conversation_end and len(conversation_history) < max_turn:
            response: Dict[str, Any] = self.agent_chain.run(user_query)

            conversation_history.append(("assistant", response))
            agent_message = response['message']
            print_with_color(f">> Assistant: {agent_message}", Fore.GREEN)

            conversation_end = self.determine_if_conversation_ends(agent_message)
            if not conversation_end:
                user_query = self.get_next_user_query(conversation_history,
                                                      test_case.user_context)
                conversation_history.append(("user", {"message": user_query}))
                print_with_color(f">> User: {user_query}", Fore.GREEN)

        is_agent_helpful = self.determine_if_agent_solved_problem(conversation_history,
                                                                  test_case.expected_outcome)
        return conversation_history, is_agent_helpful, response

    def run_test(self, test):
        agent = test.agent_cls.from_llm_and_tools(
            self.llm, test.tools, policy_desp=test.policy
        )
        self.agent_chain = Chain(tools=test.tools, agent=agent, memory=self.memory)

        test_results = []
        for i, test_case in enumerate(test.test_cases):
            conversation_history, is_agent_helpful, last_response = self.test_each_case(test_case)
            test_results.append({
                "test_name": test_case.test_name,
                "conversation_history": [f"{user_type}: {message}" for user_type, message,
                                         in conversation_history],
                "is_agent_helpful": is_agent_helpful,
                "actions_took": [{
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": action.observation
                } for action in last_response[constants.INTERMEDIATE_STEPS]],
                "num_turns": len(conversation_history),
                "expected_outcome": test_case.expected_outcome
            })

        df = pd.DataFrame(test_results)
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_json(os.path.join(self.output_dir, f"{test.__class__.__name__}.jsonl"), lines=True,
                   orient="records")

    def run_all_tests(self):
        for test in self.tests:
            self.run_test(test)

    def run_interactive(self):
        self.memory.clear()
        test = self.tests[0]
        agent = test.agent_cls.from_llm_and_tools(
            self.llm, test.tools, policy_desp=test.policy
        )
        self.agent_chain = Chain(tools=test.tools, agent=agent, memory=self.memory)

        while True:
            user_query = input(">> User: ")
            response = self.agent_chain.run(user_query)['message']
            print_with_color(response, Fore.GREEN)

    def determine_if_conversation_ends(self, last_utterance: str) -> bool:
        messages = [
            UserMessage(content=f"""The most recent reply from assistant
assistant: "{last_utterance}"
Is assistant asking a clarifying question or getting additional information from user? answer with 
yes or no"""),
        ]
        output: Generation = self.llm.generate(messages=messages).generations[0]

        if 'yes' in output.message.content.lower():
            # this is a clarifying question
            return False
        else:
            # conversation should end
            return True

    def get_next_user_query(self,
                            conversation_history: List[Tuple[str, str]],
                            user_context: str) -> str:
        messages = []
        conversation = ""
        for user_type, utterance in conversation_history:
            conversation += f"{user_type}: {utterance}\n"

        messages.append(
            UserMessage(content=f"""You are a customer with access to the following context information about yourself. 
Please respond to assistant question and try to resolve your problems in english sentence. 
If you are not sure about how to answer, respond with "hand off to agent".
Context:
"{user_context}"

Previous conversation:
{conversation}"""))

        output: Generation = self.llm.generate(messages=messages, stop=["."]).generations[0]
        return output.message.content

    def determine_if_agent_solved_problem(self,
                                          conversation_history: List[Tuple[str, str]],
                                          expected_outcome: str) -> (bool, str):
        messages = []
        conversation = ""
        for user_type, utterance in conversation_history:
            conversation += f"{user_type}: {utterance}\n"

        messages.append(
            UserMessage(content=f"""Previous conversation:
{conversation}

Expected outcome is {expected_outcome}
Does conversation reach the expected outcome for user? Answer with yes or no with explanation"""))

        output: Generation = self.llm.generate(messages=messages, stop=["."]).generations[0]
        if 'yes' in output.message.content.lower():
            # Agent solved the problem
            return True, output.message.content
        else:
            # did not solve the problem
            return False, output.message.content
