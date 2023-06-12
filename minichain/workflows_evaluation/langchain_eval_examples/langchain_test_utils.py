from typing import List

from langchain.agents import AgentType, initialize_agent
from langchain.base_language import BaseLanguageModel as LCBaseLanguageModel
from langchain.chat_models import ChatOpenAI as LangchainModel
from langchain.memory import ConversationBufferMemory as LCConversationBufferMemory
from langchain.schema import BaseMemory as LCBaseMemory
from langchain.tools import Tool as LCTool

from minichain.chain.langchain_wapper_chain import LangChainWrapperChain


def create_langchain_from_test(
    tools: List[LCTool],
    agent_type: AgentType,
    memory: LCBaseMemory = None,
    llm: LCBaseLanguageModel = None,
):
    """
    Create LangChainWrapperChain by instantiating LangChain agent
    Args:
        tools: list of langchain tool
        agent_type: LangChain AgentType
        memory: LangChain memory
        llm: LangChain language model

    Returns:
        LangChainWrapperChain
    """
    llm = llm or LangchainModel(temperature=0)
    memory = memory or LCConversationBufferMemory(memory_key="chat_history")

    langchain = initialize_agent(
        tools, llm, agent=agent_type, verbose=True, memory=memory
    )

    return LangChainWrapperChain(langchain=langchain)
