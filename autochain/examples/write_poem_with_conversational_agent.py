from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.utils import get_args

# Set logging level
_ = get_args()

llm = ChatOpenAI(temperature=0)
memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm)
chain = Chain(agent=agent, memory=memory)

user_query = "Write me a poem about AI"
print(f">> User: {user_query}")
print(
    f""">>> Assistant: 
{chain.run(user_query)["message"]}
"""
)
