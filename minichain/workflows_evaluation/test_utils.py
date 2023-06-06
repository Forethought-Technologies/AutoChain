import argparse
import logging

from minichain.agent.conversational_agent.conversational_agent import ConversationalAgent
from minichain.chain.chain import Chain
from minichain.memory.base import BaseMemory
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.base import BaseLanguageModel
from minichain.models.chat_openai import ChatOpenAI
from minichain.workflows_evaluation.base_test import BaseTest


def get_test_args():
    """Adding arguments for running test"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interact",
        "-i",
        action="store_true",
        help="if run interactively",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if show detailed contents, such as intermediate results and prompts",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    return args


def create_chain_from_test(test: BaseTest, memory: BaseMemory = None,
                           llm: BaseLanguageModel = None,
                           agent_cls=ConversationalAgent,
                           **kwargs):
    """
    Create Chain for running tests
    Args:
        test: instance of BaseTest
        memory: memory store for chain
        llm: model for agent
        agent_cls: metadata class for instantiating agent
    Returns:
        Chain
    """
    llm = llm or ChatOpenAI(temperature=0)
    memory = memory or BufferMemory()
    agent = agent_cls.from_llm_and_tools(
        llm=llm, tools=test.tools, **kwargs
    )
    return Chain(tools=test.tools, agent=agent, memory=memory)
