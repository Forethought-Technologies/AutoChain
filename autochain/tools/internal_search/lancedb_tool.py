from typing import List, Any
from dataclasses import dataclass

import lancedb
import pandas as pd

from autochain.tools.base import Tool
from autochain.models.base import BaseLanguageModel
from autochain.tools.internal_search.base_search_tool import BaseSearchTool

@dataclass
class LanceDBDoc:
    doc: str
    vector: List[float] = None

class LanceDBSeach(Tool, BaseSearchTool):
    """
    Use LanceDB as the internal search tool

    LanceDB is a vector database that supports vector search.

    Args:
        uri: the uri of the database. Default to "lancedb"
        table_name: the name of the table. Default to "table"
        metric: the metric used for vector search. Default to "cosine"
        encoder: the encoder used to encode the documents. Default to None
        docs: the documents to be indexed. Default to None
    """
    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    docs: List[LanceDBDoc]
    uri: str = "lancedb"
    table_name: str = "table"
    metric: str = "cosine"
    encoder: BaseLanguageModel = None
    db: lancedb.db.DBConnection = None
    table: lancedb.table.Table = None
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.db = lancedb.connect(self.uri)
        if self.docs:
            self._encode_docs(self.docs)
            self._create_table(self.docs)
    
    def _create_table(self, docs: List[LanceDBDoc]) -> None:
        self.table = self.db.create_table(self.table_name, self._docs_to_dataframe(docs), mode="overwrite")

    def _encode_docs(self, docs: List[LanceDBDoc]) -> None:
        for doc in docs:
            if not doc.vector:
                if not self.encoder:
                    raise ValueError("Encoder is not provided for encoding docs")
                doc.vector = self.encoder.encode([doc.doc]).embeddings[0]
    
    def _docs_to_dataframe(self, docs: List[LanceDBDoc]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"doc": doc.doc, "vector": doc.vector}
                for doc in docs
            ]
        )
    
    def _run(
        self,
        query: str,
        top_k: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if self.table is None:
            return ""

        embeddings = self.encoder.encode([query]).embeddings[0]
        result = self.table.search(embeddings).limit(top_k).to_df()["doc"].to_list()

        return  "\n".join([f"Doc {i}: {doc}" for i, doc in enumerate(result)])

    def add_docs(self, docs: List[LanceDBDoc], **kwargs):
        if not len(docs):
            return

        self._encode_docs(docs)
        self.table.add(self._docs_to_dataframe(docs)) if self.table else self._create_table(docs)
    
    def clear_index(self):
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self.table = None
