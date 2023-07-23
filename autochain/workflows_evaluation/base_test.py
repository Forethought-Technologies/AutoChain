import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import pandas as pd
from colorama import Fore

from autochain.agent.message import UserMessage
from autochain.chain import constants
from autochain.chain.base_chain import BaseChain
from autochain.models.base import Generation
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool
from autochain.utils import print_with_color
from autochain.workflows_evaluation.test_utils import parse_evaluation_response


@dataclass
class TestCase:
    """Standardized data class for each test case for BastTest"""

    test_name: str = ""
    user_context: str = ""
    expected_outcome: str = ""


class BaseTest(ABC):
    @property
    @abstractmethod
    def chain(self) -> BaseChain:
        """Chain to test with, which support run(<user_query>) function"""

    @property
    @abstractmethod
    def tools(self) -> List[Tool]:
        """Workflow policy"""

    @property
    @abstractmethod
    def test_cases(self) -> List[TestCase]:
        """"""


class WorkflowTester:
    def __init__(self, tests: List[BaseTest], output_dir: str):
        self.chain = None
        self.tests = tests
        self.output_dir = output_dir
        self.llm = ChatOpenAI(temperature=0)

    def test_each_case(self, test_case: TestCase):
        self.chain.memory.clear()

        conversation_history = []
        user_query = ""
        conversation_end = False
        max_turn = 8
        response = {}
        while not conversation_end and len(conversation_history) < max_turn:
            if not conversation_end:
                user_query = self.get_next_user_query(
                    conversation_history, test_case.user_context
                )
                conversation_history.append(("user", user_query))
                print_with_color(f">> User: {user_query}", Fore.GREEN)

            response: Dict[str, Any] = self.chain.run(user_query)

            agent_message = response["message"]
            conversation_history.append(("assistant", agent_message))
            print_with_color(f">> Assistant: {agent_message}", Fore.GREEN)

            conversation_end = self.determine_if_conversation_ends(agent_message)

        is_agent_helpful = self.determine_if_agent_solved_problem(
            conversation_history, test_case.expected_outcome
        )
        return conversation_history, is_agent_helpful, response

    def run_test(self, test):
        test_results = []
        self.chain = test.chain
        for i, test_case in enumerate(test.test_cases):
            print(
                f"========== Start running test case: {test_case.test_name} ==========\n"
            )
            conversation_history, is_agent_helpful, last_response = self.test_each_case(
                test_case
            )
            test_results.append(
                {
                    "test_name": test_case.test_name,
                    "conversation_history": [
                        f"{user_type}: {message}"
                        for user_type, message, in conversation_history
                    ],
                    "num_turns": len(conversation_history),
                    "expected_outcome": test_case.expected_outcome,
                    "is_agent_helpful": is_agent_helpful,
                    "actions_took": [
                        {
                            "tool": action.tool,
                            "tool_input": action.tool_input,
                            "tool_output": action.tool_output,
                        }
                        for action in last_response[constants.INTERMEDIATE_STEPS]
                    ],
                }
            )

        df = pd.DataFrame(test_results)
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_json(
            os.path.join(self.output_dir, f"{test.__class__.__name__}.jsonl"),
            lines=True,
            orient="records",
        )

    def run_all_tests(self):
        for test in self.tests:
            self.run_test(test)

    def run_interactive(self):
        test = self.tests[0]
        self.chain = test.chain
        self.chain.memory.clear()

        while True:
            user_query = input(">> User: ")
            response = self.chain.run(user_query)["message"]
            print_with_color(f">> Assistant: {response}", Fore.GREEN)

    def determine_if_conversation_ends(self, last_utterance: str) -> bool:
        messages = [
            UserMessage(
                content=f"""The most recent reply from assistant
assistant: "{last_utterance}"
Has assistant finish assisting the user or tries to hand off to an agent? Answer with yes or no"""
            ),
        ]
        output: Generation = self.llm.generate(messages=messages).generations[0]

        if "yes" in output.message.content.lower():
            # finish assisting; conversation should end
            return True
        else:
            # not yet finished; conversation should continue
            return False

    def get_next_user_query(
        self, conversation_history: List[Tuple[str, str]], user_context: str
    ) -> str:
        messages = []
        conversation = ""

        for user_type, utterance in conversation_history:
            conversation += f"{user_type}: {utterance}\n"

        conversation += "user: "

        messages.append(
            UserMessage(
                content=f"""You are a user with access to the following context information about yourself. 
Based on previous conversation, write the message to assistant to help you with goal described 
in context without asking repetitive questions.
Replies 'Thank you' if the goal is achieved.
If you are not sure about how to answer, respond with "hand off to agent".
Context:
"{user_context}"

Previous conversation:
{conversation}"""
            )
        )

        output: Generation = self.llm.generate(
            messages=messages, stop=[".", "?"]
        ).generations[0]
        return output.message.content

    def determine_if_agent_solved_problem(
        self, conversation_history: List[Tuple[str, str]], expected_outcome: str
    ) -> Dict[str, str]:
        messages = []
        conversation = ""
        for user_type, utterance in conversation_history:
            conversation += f"{user_type}: {utterance}\n"

        messages.append(
            UserMessage(
                content=f"""You are an admin for assistant and check if assistant meets the expected outcome based on previous conversation.
 
Previous conversation:
{conversation}

Expected outcome is "{expected_outcome}"
Does conversation reach the expected outcome for user? answer in JSON format
{{
    "reason": "explain step by step if conversation reaches the expected outcome",
    "rating": "rating from 1 to 5; 1 for not meeting the expected outcome at all, 5 for completely meeting the expected outcome",
}}"""
            )
        )

        output: Generation = self.llm.generate(messages=messages).generations[0]
        return parse_evaluation_response(output.message)
