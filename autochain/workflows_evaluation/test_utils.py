from typing import List, Optional, Dict

from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)

from autochain.agent.structs import AgentOutputParser
from autochain.agent.message import BaseMessage
from autochain.chain.chain import Chain
from autochain.memory.base import BaseMemory
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.base import BaseLanguageModel
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool


def create_chain_from_test(
    tools: List[Tool],
    memory: Optional[BaseMemory] = None,
    llm: Optional[BaseLanguageModel] = None,
    agent_cls=ConversationalAgent,
    **kwargs
):
    """
    Create Chain for running tests
    Args:
        tools: list of autochain tools
        memory: memory store for chain
        llm: model for agent
        agent_cls: metadata class for instantiating agent
    Returns:
        Chain
    """
    llm = llm or ChatOpenAI(temperature=0)
    memory = memory or BufferMemory()
    agent = agent_cls.from_llm_and_tools(llm=llm, tools=tools, **kwargs)
    return Chain(agent=agent, memory=memory)


def parse_evaluation_response(message: BaseMessage) -> Dict[str, str]:
    """
    Parse the reason and rating from the call to determine if the conversation reaches the
    expected outcome
    """
    response = AgentOutputParser.load_json_output(message)
    return {
        "rating": response.get("rating"),
        "reason": response.get("reason"),
    }
