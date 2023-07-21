from unittest import mock

import pytest

from autochain.agent.message import (
    ChatMessageHistory,
    MessageType,
)
from autochain.agent.openai_functions_agent.openai_functions_agent import (
    OpenAIFunctionsAgent,
)
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.models.chat_openai import ChatOpenAI


@pytest.fixture
def openai_function_calling_fixture():
    with mock.patch(
        "autochain.models.chat_openai.ChatOpenAI.generate_with_retry",
        return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "get_current_weather",
                            "arguments": '{\n  "location": "Toronto, Canada",\n  "format": "celsius"\n}',
                        },
                    }
                }
            ],
            "usage": 10,
        },
    ):
        yield


@pytest.fixture
def openai_response_fixture():
    with mock.patch(
        "autochain.models.chat_openai.ChatOpenAI.generate_with_retry",
        return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Sure, let me get that information for you.",
                    }
                }
            ],
            "usage": 10,
        },
    ):
        yield


def test_function_calling_plan(openai_function_calling_fixture):
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=ChatOpenAI(), tools=[])

    history = ChatMessageHistory()
    history.save_message("first user query", MessageType.UserMessage)
    history.save_message("assistant response", MessageType.AIMessage)
    history.save_message("second user query", MessageType.UserMessage)

    action = agent.plan(history=history, intermediate_steps=[])
    assert isinstance(action, AgentAction)
    assert action.tool == "get_current_weather"


def test_response_plan(openai_response_fixture):
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=ChatOpenAI(), tools=[])

    history = ChatMessageHistory()
    history.save_message("first user query", MessageType.UserMessage)
    history.save_message("assistant response", MessageType.AIMessage)
    history.save_message("second user query", MessageType.UserMessage)

    action = agent.plan(history=history, intermediate_steps=[])
    assert isinstance(action, AgentFinish)
