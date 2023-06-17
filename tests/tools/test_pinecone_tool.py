from typing import List, Optional
from unittest import mock

import pytest

from autochain.agent.message import BaseMessage
from autochain.models.base import BaseLanguageModel, LLMResult, EmbeddingResult
from autochain.tools.base import Tool
from autochain.tools.internal_search.pinecone_tool import PineconeSearch, PineconeDoc


class MockIndex:
    def upsert(self, *args, **kwargs):
        return None

    def query(self, *args, **kwargs):
        return {
            "matches": [
                {
                    "id": "A",
                    "score": 0.9,
                }
            ],
            "namespace": "",
        }


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
def pinecone_search_fixture():
    with mock.patch(
        "pinecone.create_index",
        return_value=None,
    ):
        yield


@pytest.fixture
def pinecone_index_fixture():
    with mock.patch(
        "pinecone.Index",
        return_value=MockIndex(),
    ):
        yield


def test_pinecone_search(pinecone_search_fixture, pinecone_index_fixture):
    docs = [PineconeDoc(doc="test_document", id="A")]

    pinecone_search = PineconeSearch(
        name="pinecone_search",
        description="internal search with pinecone",
        docs=docs,
        encoder=DummyEncoder(),
    )
    assert pinecone_search.docs[0].vector == [
        -0.025949304923415184,
        -0.012664584442973137,
        0.017791053280234337,
    ]
    assert pinecone_search.run({"query": "test question"}) == "Doc 0: test_document"
