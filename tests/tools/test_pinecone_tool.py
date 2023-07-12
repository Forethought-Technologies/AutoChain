from autochain.tools.internal_search.pinecone_tool import PineconeSearch, PineconeDoc
from test_utils.pinecone_mocks import (
    DummyEncoder,
    pinecone_index_fixture,
)


def test_pinecone_search(pinecone_index_fixture):
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
