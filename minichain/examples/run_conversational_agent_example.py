from minichain.chain.chain import Chain
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.chat_openai import ChatOpenAI
from minichain.tools.base import Tool
from minichain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)

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
