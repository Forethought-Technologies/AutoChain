import json

from autochain.agent.openai_functions_agent.openai_functions_agent import (
    OpenAIFunctionsAgent,
)
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool
from autochain.utils import get_args

# Set logging level
_ = get_args()


def get_current_weather(location: str, unit: str = "fahrenheit"):
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
        name="get_current_weather",
        func=get_current_weather,
        description="""Get the current weather in a given location""",
    )
]

memory = BufferMemory()
llm = ChatOpenAI(temperature=0)
agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)

# example
user_query = "What's the weather today?"
print(f">> User: {user_query}")
print(f">> Assistant: {chain.run(user_query)['message']}")
next_user_query = "Boston"
print(f">> User: {next_user_query}")
print(f">> Assistant: {chain.run(next_user_query)['message']}")
