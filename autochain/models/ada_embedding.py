import os
from typing import List, Optional, Any, Dict

from pydantic import root_validator

from autochain.tools.base import Tool

from autochain.agent.message import BaseMessage

from autochain.models.base import BaseLanguageModel, LLMResult, EmbeddingResult


class OpenAIAdaEncoder(BaseLanguageModel):
    """
    Text encoder using OpenAI Model
    """

    client: Any  #: :meta private:
    model_name: str = "text-embedding-ada-002"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = os.environ["OPENAI_API_KEY"]
        try:
            import openai

        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        openai.api_key = openai_api_key
        try:
            values["client"] = openai.Embedding
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        return values

    def generate(
        self,
        messages: List[BaseMessage],
        functions: Optional[List[Tool]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        pass

    def encode(self, texts: List[str]) -> EmbeddingResult:
        def _format_response(texts, resp) -> EmbeddingResult:
            embeddings = [d.get("embedding") for d in resp.get("data", [])]
            return EmbeddingResult(texts=texts, embeddings=embeddings)

        params: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            **self._default_params,
        }

        response = self.generate_with_retry(**params)
        return _format_response(texts=texts, resp=response)
