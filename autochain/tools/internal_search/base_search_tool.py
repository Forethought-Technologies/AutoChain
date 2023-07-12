from abc import abstractmethod
from typing import Any, List

from pydantic import Extra, BaseModel


class BaseSearchTool(BaseModel):
    class Config:
        """Configuration for this pydantic object."""

    extra = Extra.forbid
    arbitrary_types_allowed = True

    @abstractmethod
    def _run(
        self,
        query: str,
        top_k: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def add_docs(self, docs: List[Any], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def clear_index(self):
        raise NotImplementedError
