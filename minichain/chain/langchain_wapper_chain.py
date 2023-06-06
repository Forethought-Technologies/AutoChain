"""Base interface that all chains should implement."""
from typing import Any, Dict

from langchain.chains.base import Chain as LangChain

from minichain.agent.structs import AgentFinish


class LangChainWrapperChain:
    langchain: LangChain = None
    memory = None

    def __init__(self, langchain: LangChain):
        self.langchain = langchain
        self.memory = self.langchain.memory

    def run(
        self,
        user_query: str,
        return_only_outputs: bool = False,
    ) -> Dict[str, Any]:
        response_msg: str = self.langchain.run(user_query)
        agent_finish = AgentFinish(message=response_msg, log="")
        return agent_finish.format_output()
