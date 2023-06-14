import json

from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


tools = [
    Tool(
        name="Get weather",
        func=get_current_weather,
        description="""Get the current weather in a given location""",
    )
]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)

user_query = "what is the weather today"
print(f">> User: {user_query}")
print(f">> Assistant: {chain.run(user_query)['message']}")

next_user_query = "What is the weather like in boston?"
print(f">> User: {next_user_query}")
print(f">> Assistant: {chain.run(next_user_query)['message']}")
