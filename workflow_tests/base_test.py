import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, Generation
from minichain.agent.support_agent import SupportAgent
from minichain.chain.chain import DefaultChain
from minichain.memory.buffer_memory import BufferMemory
from minichain.tools.base import Tool


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
        return SupportAgent

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
        print(f">> User: {test_case.user_query}")

        conversation_end = False
        max_turn = 8
        while not conversation_end and len(conversation_history) < max_turn:
            response = self.agent_chain.run(user_query)
            conversation_history.append(("assistant", response))
            print(f">> Assistant: {response}")

            conversation_end = self.determine_if_conversation_ends(response)
            if not conversation_end:
                user_query = self.get_next_user_query(conversation_history,
                                                      test_case.user_context)
                conversation_history.append(("user", user_query))
                print(f">> User: {user_query}")

        is_agent_helpful = self.determine_if_agent_solved_problem(conversation_history,
                                                                  test_case.expected_outcome)
        return conversation_history, is_agent_helpful

    def run_test(self, test):
        agent = test.agent_cls.from_llm_and_tools(
            self.llm, test.tools, policy_desp=test.policy
        )
        self.agent_chain = DefaultChain(tools=test.tools, agent=agent, memory=self.memory)

        test_results = []
        for i, test_case in enumerate(test.test_cases):
            conversation_history, is_agent_help = self.test_each_case(test_case)
            test_results.append({
                "test_name": test_case.test_name,
                "conversation_history": [f"{user_type}: {utterance}" for user_type, utterance, in
                                         conversation_history],
                "is_agent_helpful": is_agent_help,
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

    def determine_if_conversation_ends(self, last_utterance: str) -> bool:
        messages = [
            [
                HumanMessage(content=f"""The most recent reply from assistant
assistant: "{last_utterance}"
Is assistant asking a clarifying question or getting additional information from user? answer with 
yes or no"""),
            ]
        ]
        output: Generation = self.llm.generate(messages=messages).generations[0][0]

        if 'yes' in output.text.lower():
            # this is a clarifying question
            return False
        else:
            # conversation should end
            return True

    def get_next_user_query(self,
                            conversation_history: List[Tuple[str, str]],
                            user_context: str) -> str:
        messages = [[]]
        conversation = ""
        for user_type, utterance in conversation_history:
            conversation += f"{user_type}: {utterance}\n"

        messages[0].append(
            HumanMessage(content=f"""You are a customer with access to the following context information about yourself. 
Please response assistant question and try to resolve your problems in english sentence. 
If you are not sure about how to answer, respond with "hand off to agent".
Context:
{user_context}

Previous conversation:
{conversation}"""))

        output: Generation = self.llm.generate(messages=messages, stop=["."]).generations[0][0]
        return output.text

    def determine_if_agent_solved_problem(self,
                                          conversation_history: List[Tuple[str, str]],
                                          expected_outcome: str) -> (bool, str):
        messages = [[]]
        conversation = ""
        for user_type, utterance in conversation_history:
            conversation += f"{user_type}: {utterance}\n"

        messages[0].append(
            HumanMessage(content=f"""Previous conversation:
{conversation}

Expected outcome is {expected_outcome}
Does conversation reach the expected outcome for user? Answer with yes or no with explanation"""))

        output: Generation = self.llm.generate(messages=messages, stop=["."]).generations[0][0]
        if 'yes' in output.text.lower():
            # Agent solved the problem
            return True, output.text
        else:
            # did not solve the problem
            return False, output.text
