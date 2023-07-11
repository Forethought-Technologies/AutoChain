import uuid
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional

import chromadb
from chromadb.api import QueryResult
from pydantic import Extra

from autochain.tools.base import Tool
from autochain.tools.internal_search.base_search_tool import BaseSearchTool


@dataclass
class ChromaDoc:
    doc: str
    metadata: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid1()))


class ChromaDBSearch(Tool, BaseSearchTool):
    """
    Use ChromaDB as internal search tool
    """

    collection_name: str = "index"
    collection: Optional[Any] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(self, docs: List[ChromaDoc], **kwargs):
        super().__init__(**kwargs)
        client = chromadb.Client()

        collection = client.create_collection(self.collection_name)
        self.collection = collection

        # Add docs to the collection. Can also update and delete. Row-based API coming soon!
        self.add_docs(docs=docs)

    def _run(
        self,
        query: str,
        top_k: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        def _format_output(query_result: QueryResult) -> str:
            """Only return the document since they are likely to be passed to prompt"""
            documents = query_result.get("documents", [])
            if len(documents) == 0:
                return ""

            docs = documents[0]
            return "\n".join([f"Doc {i}: {doc}" for i, doc in enumerate(docs)])

        result = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        return _format_output(result)

    def add_docs(self, docs: List[ChromaDoc], **kwargs):
        """Add a list of documents to collection"""
        if docs:
            self.collection.add(
                documents=[d.doc for d in docs],
                # we embed for you, or bring your own
                metadatas=[d.metadata for d in docs],
                # filter on arbitrary metadata!
                ids=[d.id for d in docs],  # must be unique for each doc
            )

    def clear_index(self):
        self.collection.delete()
