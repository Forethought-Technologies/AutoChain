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


@pytest.fixture
def openai_estimate_confidence_fixture():
    with mock.patch(
        "autochain.models.chat_openai.ChatOpenAI.generate_with_retry",
        return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "the confidence is 4.",
                    }
                }
            ],
            "usage": 10,
        },
    ):
        yield


@pytest.fixture
def is_generation_confident_fixture():
    with mock.patch(
        "autochain.agent.openai_functions_agent.openai_functions_agent.OpenAIFunctionsAgent.is_generation_confident",
        return_value=True,
    ):
        yield


def test_function_calling_plan(
    openai_function_calling_fixture, is_generation_confident_fixture
):
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=ChatOpenAI(), tools=[])

    history = ChatMessageHistory()
    history.save_message("first user query", MessageType.UserMessage)
    history.save_message("assistant response", MessageType.AIMessage)
    history.save_message("second user query", MessageType.UserMessage)

    action = agent.plan(history=history, intermediate_steps=[])
    assert isinstance(action, AgentAction)
    assert action.tool == "get_current_weather"


def test_response_plan(openai_response_fixture, is_generation_confident_fixture):
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=ChatOpenAI(), tools=[])

    history = ChatMessageHistory()
    history.save_message("first user query", MessageType.UserMessage)
    history.save_message("assistant response", MessageType.AIMessage)
    history.save_message("second user query", MessageType.UserMessage)

    action = agent.plan(history=history, intermediate_steps=[])
    assert isinstance(action, AgentFinish)


def test_estimate_confidence(openai_estimate_confidence_fixture):
    agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=ChatOpenAI(), tools=[])

    history = ChatMessageHistory()
    history.save_message("first user query", MessageType.UserMessage)
    history.save_message("assistant response", MessageType.AIMessage)
    history.save_message("second user query", MessageType.UserMessage)

    agent_finish_output = AgentFinish(message="agent response", log="sample log")
    is_confident = agent.is_generation_confident(
        history=history, agent_output=agent_finish_output, min_confidence=3
    )
    assert is_confident

    is_confident = agent.is_generation_confident(
        history=history, agent_output=agent_finish_output, min_confidence=5
    )
    assert is_confident is False

    agent_action_output = AgentAction(
        tool="get_current_weather",
        tool_input={"location": "Toronto, Canada", "format": "celsius"},
    )
    is_confident = agent.is_generation_confident(
        history=history, agent_output=agent_action_output, min_confidence=3
    )

    assert is_confident
