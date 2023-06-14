import json
from typing import List, Optional

from minichain.agent.support_agent.support_agent import SupportAgent
from minichain.agent.message import BaseMessage, AIMessage
from minichain.agent.structs import AgentFinish
from minichain.models.base import LLMResult, Generation, BaseLanguageModel
from minichain.tools.simple_handoff.tool import HandOffToAgent


class MockLLM(BaseLanguageModel):
    message: str = ""

    def generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> LLMResult:
        return LLMResult(
            generations=[Generation(message=AIMessage(content=self.message))]
        )


def test_should_answer_prompt():
    agent = SupportAgent.from_llm_and_tools(
        llm=MockLLM(message="yes, question is resolved"), tools=[]
    )

    input = {"query": "user query", "history": "conversation history"}
    response = agent.should_answer(**input)
    assert isinstance(response, AgentFinish)

    agent = SupportAgent(llm=MockLLM(message="no, question is not resolved"))
    response = agent.should_answer(**input)
    assert response is None


def test_plan():
    mock_generation_response = json.dumps(
        {
            "thoughts": {
                "plan": "Given workflow policy and previous observations",
                "need_use_tool": "Yes if needs to use another tool not used in previous observations else No",
            },
            "tool": {"name": "", "args": {"arg_name": ""}},
            "response": "response to suer",
            "workflow_finished": "No",
        }
    )

    agent = SupportAgent.from_llm_and_tools(
        llm=MockLLM(message=mock_generation_response), tools=[HandOffToAgent()]
    )

    input = {"query": "user query", "history": "conversation history"}
    action = agent.plan(intermediate_steps=[], **input)
    assert isinstance(action, AgentFinish)
