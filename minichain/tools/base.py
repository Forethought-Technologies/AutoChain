"""Base implementation for tools or skills."""
from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from minichain.errors import ToolRunningError
from pydantic import (
    BaseModel,
)


class Tool(ABC, BaseModel):
    """Interface MiniChain tools must implement."""

    name: str
    """The unique name of the tool that clearly communicates its purpose."""
    description: str
    """Used to tell the model how/when/why to use the tool.

    You can provide few-shot examples as a part of the description.
    """
    args_schema: Optional[Type[BaseModel]] = None
    """Pydantic model class to validate and parse the tool's input arguments."""
    return_direct: bool = False
    """Whether to return the tool's output directly. Setting this to True means

    that after the tool is called, the AgentExecutor will stop looping.
    """
    verbose: bool = False
    """Whether to log the tool's progress."""

    func: Union[Callable[..., str], None] = None

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
    ) -> Any:
        return self.func(*args, **kwargs)

    def run(
        self,
        tool_input: Union[str, Dict] = "",
        **kwargs: Any,
    ) -> Any:
        """Run the tool."""
        try:
            parsed_input = self._parse_input(tool_input)
        except ValueError as e:
            # return exception as observation
            raise ToolRunningError(message=f"Tool input args value Error: {e}") from e

        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
            observation = self._run(*tool_args, **tool_kwargs)
        except (Exception, KeyboardInterrupt) as e:
            raise ToolRunningError(
                message=f"Failed to run tool {self.name} due to {e}"
            ) from e

        return observation
