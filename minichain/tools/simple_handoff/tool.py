from typing import Any

from minichain.tools.base import Tool


class HandOffToAgent(Tool):
    name = "Hand off"
    description = "Hand off to a customer support agent"
    handoff_msg = "Let me hand you off to an agent now"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return self.handoff_msg
