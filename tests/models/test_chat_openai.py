import os
from unittest import mock

import pytest

from minichain.agent.message import UserMessage
from minichain.models.base import LLMResult
from minichain.models.chat_openai import ChatOpenAI


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
