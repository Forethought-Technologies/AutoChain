import os
from unittest import mock

import pytest

from autochain.models.ada_embedding import OpenAIAdaEncoder
from autochain.models.base import EmbeddingResult


@pytest.fixture
def ada_encoding_fixture():
    with mock.patch(
        "openai.Embedding.create",
        return_value={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [
                        -0.025949304923415184,
                        -0.012664584442973137,
                        0.017791053280234337,
                    ],
                }
            ],
            "model": "text-embedding-ada-002-v2",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    ):
        yield


def test_ada_encoder(ada_encoding_fixture):
    text = "example text"
    os.environ["OPENAI_API_KEY"] = "mock_api_key"

    encoder = OpenAIAdaEncoder(temperature=0)
    response = encoder.encode([text])

    assert response
    assert isinstance(response, EmbeddingResult)
    assert response.texts[0] == text
    assert len(response.embeddings[0]) > 0
    assert response.embeddings[0] == [
        -0.025949304923415184,
        -0.012664584442973137,
        0.017791053280234337,
    ]
