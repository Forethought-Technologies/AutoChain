from autochain.tools.internal_search.chromadb_tool import ChromaDBSearch, ChromaDoc


def test_chromadb_tool_run():
    d1 = ChromaDoc("This is document1", metadata={"source": "notion"})

    d2 = ChromaDoc("This is document2", metadata={"source": "google-docs"})

    t = ChromaDBSearch(
        docs=[d1, d2], name="internal_search", description="internal search"
    )
    output = t.run({"query": "This is a query document", "n_results": 2})
    assert output == "Doc 0: This is document1\nDoc 1: This is document2"
