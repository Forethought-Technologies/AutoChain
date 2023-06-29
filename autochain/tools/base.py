"""Base implementation for tools or skills."""
from __future__ import annotations

import inspect
from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from autochain.errors import ToolRunningError
from pydantic import (
    BaseModel,
    root_validator,
)


class Tool(ABC, BaseModel):
    """Interface AutoChain tools must implement."""

    name: Optional[str] = None
    """The unique name of the tool that clearly communicates its purpose.
    If not provided, it will be named after the func name.
    The more descriptive it is, the easier it would be for model to call the right tool
    """

    description: str
    """Used to tell the model how/when/why to use the tool.
    You can provide few-shot examples as a part of the description.
    """

    arg_description: Optional[Dict[str, Any]] = None
    """Dictionary of arg name and description when using OpenAIFunctionsAgent to provide 
    additional argument information"""

    args_schema: Optional[Type[BaseModel]] = None
    """Pydantic model class to validate and parse the tool's input arguments."""

    func: Union[Callable[..., str], None] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        func = values.get("func")
        if func and not values.get("name"):
            values["name"] = values["func"].__name__

        # check if all args from arg_description exist in func args
        if values.get("arg_description") and func:
            inspection = inspect.getfullargspec(func)
            override_args = set(values["arg_description"].keys())
            args = set(inspection.args)
            override_without_args = override_args - args
            if len(override_without_args) > 0:
                raise ValueError(
                    f"Provide arg description for not existed args: {override_without_args}"
                )

        return values

    def _parse_input(
        self,
        tool_input: Union[str, Dict],
    ) -> Union[str, Dict[str, Any]]:
        """Convert tool input to pydantic model."""
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                key_ = next(iter(input_args.__fields__.keys()))
                input_args.validate({key_: tool_input})
            return tool_input
        else:
            if input_args is not None:
                result = input_args.parse_obj(tool_input)
                return {k: v for k, v in result.dict().items() if k in tool_input}
        return tool_input

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        # For backwards compatibility, if run_input is a string,
        # pass as a positional argument.
        if isinstance(tool_input, str):
            return (tool_input,), {}
        else:
            return (), tool_input

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        return self.func(*args, **kwargs)

    def run(
        self,
        tool_input: Union[str, Dict] = "",
        **kwargs: Any,
    ) -> str:
        """Run the tool."""
        try:
            parsed_input = self._parse_input(tool_input)
        except ValueError as e:
            # return exception as tool output
            raise ToolRunningError(message=f"Tool input args value Error: {e}") from e

        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            tool_output = self._run(*tool_args, **tool_kwargs)
        except (Exception, KeyboardInterrupt) as e:
            raise ToolRunningError(
                message=f"Failed to run tool {self.name} due to {e}"
            ) from e

        return tool_output
