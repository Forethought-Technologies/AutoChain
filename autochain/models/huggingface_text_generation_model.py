"""OpenAI chat wrapper."""
from __future__ import annotations

import enum
import inspect
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from pydantic import Extra, Field, root_validator

from autochain.agent.message import (
    BaseMessage,
    UserMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
)
from autochain.models.base import (
    LLMResult,
    Generation,
    BaseLanguageModel,
)
from autochain.tools.base import Tool

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class HuggingFaceTextGenerationModel(BaseLanguageModel):
    model_name: str = "gpt2"
    """Model name to use."""
    temperature: float = 0
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 6
    # TODO: support streaming
    # """Maximum number of retries to make when generating."""
    # streaming: bool = False
    # """Whether to stream the results or not."""
    # n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    pipeline_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Other huggingface pipeline args"""

    model: Optional[AutoModelForCausalLM]
    tokenizer: Optional[AutoTokenizer]

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.model_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **self.model_kwargs
        )

    def generate(
        self,
        messages: List[BaseMessage],
        functions: Optional[List[Tool]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs=self.model_kwargs,
            **self.pipeline_kwargs,
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

        if stop:
            text = self._enforce_stop_tokens(text=text, stop=stop)
        return LLMResult(
            generations=[Generation(message=AIMessage(content=text))],
            llm_output={
                "token_usage": len(text.split()),
                "model_name": self.model_name,
            },
        )
