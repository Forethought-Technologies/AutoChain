from langchain.agents import AgentType, initialize_agent
from langchain.base_language import BaseLanguageModel as LCBaseLanguageModel
from langchain.chat_models import ChatOpenAI as LangchainModel
from langchain.memory import ConversationBufferMemory as LCConversationBufferMemory
from langchain.schema import BaseMemory as LCBaseMemory

from minichain.chain.langchain_wapper_chain import LangChainWrapperChain
from minichain.workflows_evaluation.base_test import BaseTest


def create_langchain_from_test(test: BaseTest, agent_type: AgentType, memory: LCBaseMemory = None,
                               llm: LCBaseLanguageModel = None):
    """
    Create LangChainWrapperChain by instantiating LangChain agent
    Args:
        test: instance of BaseTest
        agent_type: LangChain AgentType
        memory: LangChain memory
        llm: LangChain language model

    Returns:
        LangChainWrapperChain
    """
    llm = llm or LangchainModel(temperature=0)
    memory = memory or LCConversationBufferMemory(memory_key="chat_history")

    langchain = initialize_agent(test.tools, llm, agent=agent_type,
                                 verbose=True, memory=memory)

    return LangChainWrapperChain(langchain=langchain)
