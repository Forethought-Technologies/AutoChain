from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI as LangchainModel
from langchain.memory import ConversationBufferMemory

from minichain.chain.langchain_wapper_chain import LangChainWrapperChain
from minichain.memory.base import BaseMemory
from minichain.models.base import BaseLanguageModel
from minichain.workflows_evaluation.base_test import BaseTest


def create_langchain_from_test(test: BaseTest, agent_type: AgentType, memory: BaseMemory = None,
                               llm: BaseLanguageModel = None):
    llm = llm or LangchainModel(temperature=0)
    memory = memory or ConversationBufferMemory(memory_key="chat_history")

    langchain = initialize_agent(test.tools, llm, agent=agent_type,
                                 verbose=True, memory=memory)

    return LangChainWrapperChain(langchain=langchain)
