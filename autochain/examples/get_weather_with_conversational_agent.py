from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.utils import get_args

# Set logging level
_ = get_args()

llm = ChatOpenAI(temperature=0)
tools = [
    Tool(
        name="Get weather",
        func=lambda *args, **kwargs: "Today is a sunny day",
        description="""This function returns the weather information""",
    )
]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)

user_query = "what is the weather today"
print(f">> User: {user_query}")
print(f">> Assistant: {chain.run(user_query)['message']}")
next_user_query = "Boston"
print(f">> User: {next_user_query}")
print(f">> Assistant: {chain.run(next_user_query)['message']}")
