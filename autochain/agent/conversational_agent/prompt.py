from __future__ import annotations

PLANNING_PROMPT_TEMPLATE = """You are an assistant who tries to have helpful conversation 
with user based on previous conversation and previous tools outputs from tools. 
${prompt}
Use tool when provided. If there is no tool available, respond with have a helpful and polite 
conversation. Find next step without using the same tool with same inputs.

Assistant has access to the following tools:
${tools}

Previous conversation so far:
${history}

Previous tools outputs:
${agent_scratchpad}

Please respond user question in JSON format as described below
RESPONSE FORMAT:
{
  "thoughts": {
    "plan": "Given previous tools outputs, what is the next step after the previous conversation",
    "need_use_tool": "answer with 'Yes' if requires more information not in previous tools outputs else 'No'"
  },
  "tool": {
    "name": "tool name, should be one of [${tool_names}] or empty if tool is not needed",
    "args": {
      "arg_name": "arg value from conversation history or tools outputs to run tool"
    }
  },
  "response": "response to user given tools outputs and conversations",
}

Ensure the response can be parsed by Python json.loads
"""

SHOULD_ANSWER_PROMPT_TEMPLATE = """You are a support agent. 
Given the following conversation so far, has assistant finish helping user with all the 
questions?
Answer with yes or no.

Conversation:
${history}
"""

FIX_TOOL_INPUT_PROMPT_TEMPLATE = """Tool have the following spec and input provided
Spec: "{tool_description}"
Inputs: "{inputs}"
Running this tool failed with the following error: "{error}"
What is the correct input in JSON format for this tool?
"""


CLARIFYING_QUESTION_PROMPT_TEMPLATE = """You are a support agent who is going to use '${tool_name}' tool.
Check if you have enough information from the previous conversation and tools outputs to use tool based on the spec below.
"${tool_desp}"

Previous conversation so far:
${history}

Previous tools outputs:
${agent_scratchpad}

Please respond user question in JSON format as described below
RESPONSE FORMAT:
{
    "has_arg_value": "Do values for all input args for '${tool_name}' tool exist? answer with Yes or No",
    "clarifying_question": "clarifying question to user to ask for missing information"
}
Ensure the response can be parsed by Python json.loads"""
