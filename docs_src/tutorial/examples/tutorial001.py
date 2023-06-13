from minichain.chain.chain import Chain
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.chat_openai import ChatOpenAI
from minichain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)

llm = ChatOpenAI(temperature=0)
memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm)
chain = Chain(agent=agent, memory=memory)

print(chain.run("Write me a poem about AI")["message"])
