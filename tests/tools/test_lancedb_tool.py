from autochain.tools.internal_search.lancedb_tool import LanceDBDoc, LanceDBSeach
from test_utils import DummyEncoder


def test_lancedb_search():
    docs = [LanceDBDoc(doc="test_document", id="A")]

    lancedb_search = LanceDBSeach(
        uri="lancedb",
        description="internal search with lancedb",
        docs=docs,
        encoder=DummyEncoder(),
    )
    assert lancedb_search.docs[0].vector == [
        -0.025949304923415184,
        -0.012664584442973137,
        0.017791053280234337,
    ]
    assert lancedb_search.run({"query": "test question"}) == "Doc 0: test_document"
