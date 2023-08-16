import json

from autochain.agent.openai_functions_agent.openai_functions_agent import (
    OpenAIFunctionsAgent,
)
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool
from autochain.workflows_evaluation.base_test import BaseTest, TestCase, WorkflowTester
from autochain.workflows_evaluation.test_utils import (
    create_chain_from_test,
)
from autochain.utils import get_args


def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


class TestGetWeatherWithFunctionCalling(BaseTest):
    prompt = """You are a weather support agent tries to get weather information for requested 
user location"""

    tools = [
        Tool(
            name="get_current_weather",
            func=get_current_weather,
            description="""Get the current weather in a given location""",
        )
    ]

    test_cases = [
        TestCase(
            test_name="get weather for boston",
            user_context="want to get current weather information; location in Boston",
            expected_outcome="found weather information in Boston",
        ),
    ]

    llm = ChatOpenAI(temperature=0)
    chain = create_chain_from_test(
        tools=tools, agent_cls=OpenAIFunctionsAgent, llm=llm, prompt=prompt
    )


if __name__ == "__main__":
    tester = WorkflowTester(
        tests=[TestGetWeatherWithFunctionCalling()],
        output_dir="./test_results",
    )

    args = get_args()
    if args.interact:
        tester.run_interactive()
    else:
        tester.run_all_tests()
