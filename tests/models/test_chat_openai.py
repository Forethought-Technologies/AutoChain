import os
from unittest import mock

import pytest
from autochain.tools.base import Tool

from autochain.agent.message import UserMessage
from autochain.models.base import LLMResult
from autochain.models.chat_openai import ChatOpenAI, convert_tool_to_dict


def sample_tool_func_no_type(k, *arg, **kwargs):
    return f"run with {k}"


def sample_tool_func_with_type(k: int, *arg, **kwargs):
    return str(k + 1)


def sample_tool_func_with_type_default(k: int, d: int = 1, *arg, **kwargs):
    return str(k + d + 1)


@pytest.fixture
def openai_completion_fixture():
    with mock.patch(
        "openai.ChatCompletion.create",
        return_value={
            "choices": [
                {"message": {"role": "assistant", "content": "generated message"}}
            ],
            "usage": 10,
        },
    ):
        yield


def test_chat_completion(openai_completion_fixture):
    os.environ["OPENAI_API_KEY"] = "mock_api_key"
    model = ChatOpenAI(temperature=0)
    response = model.generate([UserMessage(content="test message")])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert response.generations[0].message.content == "generated message"


def test_convert_tool_to_dict():
    no_type_tool = Tool(
        func=sample_tool_func_no_type,
        description="""This is just a dummy tool without typing info""",
    )

    tool_dict = convert_tool_to_dict(no_type_tool)

    assert tool_dict == {
        "name": "sample_tool_func_no_type",
        "description": "This is just a " "dummy tool without typing info",
        "parameters": {
            "type": "object",
            "properties": {"k": {"type": "string"}},
            "required": ["k"],
        },
    }

    with_type_tool = Tool(
        func=sample_tool_func_with_type,
        description="""This is just a dummy tool with typing info""",
    )

    with_type_tool_dict = convert_tool_to_dict(with_type_tool)
    assert with_type_tool_dict == {
        "name": "sample_tool_func_with_type",
        "description": "This is just a dummy tool with typing info",
        "parameters": {
            "type": "object",
            "properties": {"k": {"type": "int"}},
            "required": ["k"],
        },
    }

    with_type_default_tool = Tool(
        func=sample_tool_func_with_type_default,
        description="""This is just a dummy tool with typing info""",
    )

    with_type_default_tool_dict = convert_tool_to_dict(with_type_default_tool)
    assert with_type_default_tool_dict == {
        "name": "sample_tool_func_with_type_default",
        "description": "This is just a dummy tool with typing info",
        "parameters": {
            "type": "object",
            "properties": {"k": {"type": "int"}, "d": {"type": "int"}},
            "required": ["k"],
        },
    }

    with_type_and_desp_tool = Tool(
        func=sample_tool_func_with_type,
        description="""This is just a dummy tool with typing info""",
        arg_description={"k": "key of the arg"},
    )

    with_type_and_desp_tool_dict = convert_tool_to_dict(with_type_and_desp_tool)
    assert with_type_and_desp_tool_dict == {
        "name": "sample_tool_func_with_type",
        "description": "This is just a dummy tool with typing info",
        "parameters": {
            "type": "object",
            "properties": {"k": {"type": "int", "description": "key of the arg"}},
            "required": ["k"],
        },
    }
