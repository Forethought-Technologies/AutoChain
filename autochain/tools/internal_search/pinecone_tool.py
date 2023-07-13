import uuid
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict

import pinecone
from pinecone import QueryResponse

from autochain.models.base import BaseLanguageModel
from autochain.tools.base import Tool
from autochain.tools.internal_search.base_search_tool import BaseSearchTool


@dataclass
class PineconeDoc:
    doc: str
    vector: List[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid1()))


class PineconeSearch(Tool, BaseSearchTool):
    """
    Use Pinecone as the internal search tool
    """

    docs: List[PineconeDoc]
    index_name: str = "index"
    index: Optional[Any] = None
    dimension: int = 8
    metric: str = "euclidean"
    encoder: BaseLanguageModel = None  # such as OpenAIAdaEncoder
    id2doc: Dict[str, str] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pinecone.create_index(
            self.index_name, dimension=self.dimension, metric=self.metric
        )
        self.index = pinecone.Index(self.index_name)

        self.add_docs(self.docs)

    def _encode(self, doc: PineconeDoc) -> None:
        if not doc.vector and self.encoder:
            # TODO: encoder over batches
            doc.vector = self.encoder.encode([doc.doc]).embeddings[0]

    def _run(
        self,
        query: str,
        top_k: int = 2,
        include_values: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        def _format_output(query_response: QueryResponse) -> str:
            """Only return the document since they are likely to be passed to prompt"""
            documents = query_response.get("matches", [])
            if len(documents) == 0:
                return ""

            return "\n".join(
                [
                    f"Doc {i}: {self.id2doc[doc['id']]}"
                    for i, doc in enumerate(documents)
                ]
            )

        encoding = self.encoder.encode([query]).embeddings[0]

        response: QueryResponse = self.index.query(
            vector=encoding, top_k=top_k, include_values=include_values
        )
        return _format_output(response)

    def add_docs(self, docs: List[PineconeDoc], **kwargs):
        if not len(docs):
            return

        for doc in docs:
            self._encode(doc)
            self.id2doc[doc.id] = doc.doc

        self.index.upsert([(d.id, d.vector) for d in docs])

    def clear_index(self):
        pinecone.delete_index(self.index_name)
        pinecone.create_index(
            self.index_name, dimension=self.dimension, metric=self.metric
        )
        self.index = pinecone.Index(self.index_name)
