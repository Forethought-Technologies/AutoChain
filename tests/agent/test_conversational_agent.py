import json
import os
from unittest import mock

import pytest

from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.agent.message import (
    ChatMessageHistory,
    MessageType,
)
from autochain.agent.structs import AgentFinish

from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.simple_handoff.tool import HandOffToAgent


@pytest.fixture
def openai_should_answer_fixture():
    with mock.patch(
        "autochain.models.chat_openai.ChatOpenAI.generate_with_retry",
        side_effect=side_effect,
    ):
        yield


def side_effect(*args, **kwargs):
    message = kwargs["messages"][0]["content"]

    if "good" in message:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "yes, question is resolved",
                    }
                }
            ],
            "usage": 10,
        }
    else:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "no, question is not resolved",
                    }
                }
            ],
            "usage": 10,
        }


@pytest.fixture
def openai_response_fixture():
    with mock.patch(
        "autochain.models.chat_openai.ChatOpenAI.generate_with_retry",
        return_value={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "thoughts": {
                                    "plan": "Given workflow policy and previous tools outputs",
                                    "need_use_tool": "Yes if needs to use another tool not used in previous tools outputs else No",
                                },
                                "tool": {"name": "", "args": {"arg_name": ""}},
                                "response": "response to suer",
                                "workflow_finished": "No",
                            }
                        ),
                    }
                }
            ],
            "usage": 10,
        },
    ):
        yield


def test_should_answer_prompt(openai_should_answer_fixture):
    os.environ["OPENAI_API_KEY"] = "mock_api_key"
    agent = ConversationalAgent.from_llm_and_tools(llm=ChatOpenAI(), tools=[])

    history = ChatMessageHistory()
    history.save_message("good user query", MessageType.UserMessage)
    inputs = {"history": history}
    response = agent.should_answer(**inputs)
    assert isinstance(response, AgentFinish)

    history = ChatMessageHistory()
    history.save_message("bad user query", MessageType.UserMessage)
    inputs = {"history": history}
    agent = ConversationalAgent(llm=ChatOpenAI(), tools=[])
    response = agent.should_answer(**inputs)
    assert response is None


def test_plan(openai_response_fixture):
    os.environ["OPENAI_API_KEY"] = "mock_api_key"
    agent = ConversationalAgent.from_llm_and_tools(
        llm=ChatOpenAI(), tools=[HandOffToAgent()]
    )

    history = ChatMessageHistory()
    history.save_message("first user query", MessageType.UserMessage)
    history.save_message("assistant response", MessageType.AIMessage)
    history.save_message("second user query", MessageType.UserMessage)

    action = agent.plan(history=history, intermediate_steps=[])
    assert isinstance(action, AgentFinish)
