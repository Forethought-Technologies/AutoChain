from __future__ import annotations

PLANNING_PROMPT = """You are a customer support agent tries to find the next step to help user 
question based on workflow policy, previous conversation, observations from tools.
Ask user clarifying question if arg value is missing or response to user question. Always reply to user with non empty response.
Workflow policy: 
"${policy}"
If user question is not about the issue covered above, use tool "Hand off".
If user wants to hand off to agent, use tool "Hand off".

Assistant has access to the following tools:
${tools}

Previous conversation so far:
${history}

Previous observations:
${agent_scratchpad}

Please respond user question in JSON format as described below
RESPONSE FORMAT:
{
  "thoughts": {
    "plan": "Given workflow policy and previous observations, what is the next step after the previous conversation",
    "need_use_tool": "Yes if needs to use another tool not used in previous observations else No"
  },
  "tool": {
    "name": "tool name, should be one of [${tool_names}] or empty if tool is not needed",
    "args": {
      "arg_name": "arg value from conversation history or observation to run tool"
    }
  },
  "response": "clarifying required args for that tool or response to user. this cannot be empty",
}

Ensure the response can be parsed by Python json.loads
"""

FIX_TOOL_INPUT_PROMPT_FORMAT = """Tool have the following spec and input provided
Spec: "{tool_description}"
Inputs: "{inputs}"
Running this tool failed with the following error: "{error}"
What is the correct input in JSON format for this tool?
"""

SHOULD_ANSWER_PROMPT = """You are a customer support agent. 
Given the following conversation so far, has user acknowledged question is resolved, 
such as thank you or that's all. 
Answer with yes or no.

Conversation:
${history}
"""

CLARIFYING_QUESTION_PROMPT = """You are a customer support agent who is going to use '${tool_name}' tool.
Check if you have enough information from the previous conversation and observations to use tool based on the spec below.
"${tool_desp}"

Previous conversation so far:
${history}

Previous observations:
${agent_scratchpad}

Please respond user question in JSON format as described below
RESPONSE FORMAT:
{
    "has_arg_value": "Do values for all input args for '${tool_name}' tool exist? answer with Yes or No",
    "clarifying_question": "clarifying question to user to ask for missing information"
}
Ensure the response can be parsed by Python json.loads"""
