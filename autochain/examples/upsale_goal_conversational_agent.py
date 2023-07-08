from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.utils import get_args

# Set logging level
_ = get_args()

prompt = (
    "You are a sales agent who wants to up sale all customer inquire. Your goal is "
    "introducing more expensive options to user"
)

llm = ChatOpenAI(temperature=0)
memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, prompt=prompt)
chain = Chain(agent=agent, memory=memory)

user_query = "How much is this basic rice cooker"
print(f">>> User: {user_query}")
print(
    f""">>> Assistant: 
{chain.run("How much is this basic rice cooker")["message"]}
"""
)
