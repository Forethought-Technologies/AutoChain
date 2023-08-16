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

logger = logging.getLogger(__name__)


def convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return UserMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(
            content=_dict["content"] or "hum..",
            function_call=_dict.get("function_call", {}),
        )
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    else:
        raise ValueError(f"Unsupported role {role}")


def convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, UserMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


def convert_tool_to_dict(tool: Tool):
    """Convert tool into function parameter for openai"""
    inspection = inspect.getfullargspec(tool.func)
    arg_description = tool.arg_description or {}

    def _type_to_string(t: type) -> str:
        prog = re.compile(r"<class '(\w+)'>")
        cls = prog.findall(str(t))

        primary_type_map = {"str": "string"}

        if len(cls) > 0:
            cls_name = cls[0].split(".")[-1]
            return primary_type_map.get(cls_name, cls_name)

        if issubclass(t, enum.Enum):
            return "enum"

        return str(t)

    def _format_property(t: type, arg_desp: str):
        p = {"type": _type_to_string(t)}
        if arg_desp:
            p["description"] = arg_desp

        return p

    arg_annotations = inspection.annotations
    if arg_annotations:
        properties = {
            arg: _format_property(t, arg_description.get(arg))
            for arg, t in arg_annotations.items()
        }
    else:
        properties = {
            arg: _format_property(str, arg_description.get(arg))
            for arg in inspection.args
        }

    default_args = inspection.defaults or []
    required_args = inspection.args[: len(inspection.args) - len(default_args)]

    output = {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required_args,
        },
    }

    return output


class ChatOpenAI(BaseLanguageModel):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from autochain.models.chat_openai import ChatOpenAI
            openai = ChatOpenAI()
    """

    client: Any  #: :meta private:
    model_name: str = "gpt-3.5-turbo"
    """Model name to use."""
    temperature: float = 0
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    api_type: Optional[str] = None
    """OpenAI API type, it can be `openai` or `azure`."""
    api_base: Optional[str] = None
    """The OpenAI API base url or Azure OpenAI API base url."""
    azure_api_version: Optional[str] = None
    """Azure API version."""
    azure_deployment_name: Optional[str] = None
    """Azure deployment name."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to OpenAI completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    # TODO: support streaming
    # streaming: bool = False
    # """Whether to stream the results or not."""
    # n: int = 1
    # """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = os.environ["OPENAI_API_KEY"]
        openai_api_type = os.environ.get("OPENAI_API_TYPE", "open_ai")
        openai_api_base = os.environ.get("OPENAI_API_BASE", None)
        try:
            import openai

        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        values["api_key"] = openai.api_key = openai_api_key
        values["api_type"] = openai.api_type = openai_api_type
        if openai_api_base:
            values["api_base"] = openai.api_base = openai_api_base
        if openai_api_type == "azure":
            values["azure_api_version"] = openai.api_version = os.environ.get("OPENAI_API_VERSION", "2023-05-15")
        try:
            values["client"] = openai.ChatCompletion
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        # if values["n"] < 1:
        #     raise ValueError("n must be at least 1.")
        # if values["n"] > 1 and values["streaming"]:
        #     raise ValueError("n must be 1 when streaming.")
        return values

    def generate(
        self,
        messages: List[BaseMessage],
        functions: Optional[List[Tool]] = None,
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        message_dicts, function_dicts, params = self._create_message_dicts(
            messages, functions, stop
        )

        generation_param = {
            "messages": message_dicts,
            **params,
        }
        if len(function_dicts) > 0:
            generation_param["functions"] = function_dicts

        response = self.generate_with_retry(**generation_param)
        return self._create_llm_result(response)

    def _create_message_dicts(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[Tool]],
        stop: Optional[List[str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if self.azure_deployment_name and self.api_type == "azure":
            params["engine"] = self.azure_deployment_name
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        function_dicts = []
        if tools:
            function_dicts = [convert_tool_to_dict(t) for t in tools]
        return message_dicts, function_dicts, params

    def _create_llm_result(self, response: Mapping[str, Any]) -> LLMResult:
        generations = []
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            gen = Generation(message=message)
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        result = LLMResult(generations=generations, llm_output=llm_output)
        return result
