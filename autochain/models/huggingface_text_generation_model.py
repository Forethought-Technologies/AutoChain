"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import Field
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from autochain.agent.message import (
    BaseMessage,
    AIMessage,
)
from autochain.models.base import (
    LLMResult,
    Generation,
    BaseLanguageModel,
)
from autochain.tools.base import Tool

logger = logging.getLogger(__name__)


class HuggingFaceTextGenerationModel(BaseLanguageModel):
    """Huggingface model that supports text-generation task

    Example:
    .. code-block:: python

        from autochain.models.huggingface_text_generation_model import HuggingFaceTextGenerationModel
        llm = HuggingFaceTextGenerationModel(model_name="mosaicml/mpt-7b", model_kwargs={"trust_remote_code":True})
    """

    model_name: str = "gpt2"
    """Model name to use. GPT2 is only for demostration purpose. It does not work well for task 
    planning"""
    temperature: float = 0
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    tokenizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for creating tokenizer."""

    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 6
    # TODO: support streaming
    # """Maximum number of retries to make when generating."""
    # streaming: bool = False
    # """Whether to stream the results or not."""
    # n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = 512
    """Maximum number of tokens to generate."""

    default_stop_tokens: List[str] = ["."]
    """Model will generate tokens up to the number of max token, so it would be good to have 
    default stop token"""

    model: Optional[AutoModelForCausalLM]
    tokenizer: Optional[AutoTokenizer]

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **self.tokenizer_kwargs
        )

    def generate(
        self,
        messages: List[BaseMessage],
        functions: Optional[List[Tool]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        generator = pipeline(
            task="text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.model_kwargs,
        )

        prompt = self._construct_prompt_from_message(messages)
        generation = generator(prompt, do_sample=False)

        return self._create_llm_result(generation=generation, prompt=prompt, stop=stop)

    @staticmethod
    def _construct_prompt_from_message(messages: List[BaseMessage]):
        prompt = ""
        for msg in messages:
            prompt += msg.content
        return prompt

    @staticmethod
    def _enforce_stop_tokens(text: str, stop: List[str]) -> str:
        """Cut off the text as soon as any stop words occur."""
        first_index = len(text)
        for s in stop:
            if s in text:
                first_index = min(text.index(s), first_index)

        return text[:first_index].strip()

    def _create_llm_result(
        self, generation: List[Dict[str, Any]], prompt: str, stop: List[str]
    ) -> LLMResult:
        text = generation[0]["generated_text"][len(prompt) :]
        if self.max_tokens:
            token_ids = self.tokenizer.encode(text)[: self.max_tokens]
            text = self.tokenizer.decode(token_ids)

        # it is better to have a default stop token so model does not always generate to max
        # sequence length
        stop = stop or self.default_stop_tokens
        text = self._enforce_stop_tokens(text=text, stop=stop)

        return LLMResult(
            generations=[Generation(message=AIMessage(content=text))],
            llm_output={
                "token_usage": len(text.split()),
                "model_name": self.model_name,
            },
        )
