from minichain.tools.internal_search.chromadb_tool import ChromaDBSearch, ChromaDoc

d1 = ChromaDoc(
    "This is document1",
    metadata={"source": "notion"}
)

d2 = ChromaDoc(
    "This is document2",
    metadata={"source": "google-docs"}
)

t = ChromaDBSearch(docs=[d1, d2], name="internal_search", description="internal search")
output = t.run({"query": "This is a query document", "n_results": 2})
print(output)