from typing import List, Optional
from unittest import mock

import pytest

from autochain.agent.message import BaseMessage
from autochain.models.base import BaseLanguageModel, LLMResult, EmbeddingResult
from autochain.tools.base import Tool


class MockIndex:
    def __init__(self):
        self.kv = {}

    def upsert(self, id_vectors, *args, **kwargs):
        for id, vector in id_vectors:
            self.kv[id] = vector

    def query(self, vector, *args, **kwargs):
        for id, v in self.kv.items():
            if vector == v:
                return {
                    "matches": [
                        {
                            "id": id,
                            "score": 0.9,
                        }
                    ],
                    "namespace": "",
                }
        else:
            return {}


class DummyEncoder(BaseLanguageModel):
    def generate(
        self,
        messages: List[BaseMessage],
        functions: Optional[List[Tool]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        pass

    def encode(self, texts: List[str]) -> EmbeddingResult:
        return EmbeddingResult(
            texts=texts,
            embeddings=[
                [-0.025949304923415184, -0.012664584442973137, 0.017791053280234337]
            ],
        )


@pytest.fixture
def pinecone_index_fixture():
    with mock.patch(
        "pinecone.create_index",
        return_value=None,
    ), mock.patch(
        "pinecone.Index",
        return_value=MockIndex(),
    ), mock.patch(
        "pinecone.delete_index",
        return_value=None,
    ):
        yield
