from __future__ import annotations

PREFIX_PROMPT = """You are a customer support agent tries to find the next step to help user question based on workflow policy, previous conversation, observations from tools.
Ask user clarifying question if arg value is missing or response to user question. Always reply to user with non empty response.
Workflow policy: 
"
{policy}
"

If user question is not about the issue covered above, use tool "Hand off".
If user wants to hand off to agent, use tool "Hand off".

Assistant has access to the following tools:
"""

CODE_DESCRIPTION_PROMPT_FORMAT = """
what is the short description for the following python code
```
{code}
```
"""

SBS_INSTRUCTION_FORMAT = """Please respond user question in JSON format as described below
RESPONSE FORMAT:
{
  "thoughts": {
    "plan": "what is the next step after the previous conversation based on workflow policy and previous observations",
    "need_use_tool": "Yes if needs to use a tool not used previously else No"
  },
  "tool": {
    "name": "tool name, should be one of [${tool_names}] or empty if tool is not needed",
    "args": {
      "arg_name": "arg value from conversation history or observation to run tool"
    }
  },
  "validation": {
    "arg_valid": "are arg values valid based on input args from tools? Yes or No"
  },
  "response": "clarifying required args for that tool or response to user. this cannot be empty",
  "workflow_finished": "Yes if reach the end of workflow else No"
}

Ensure the response can be parsed by Python json.loads"""

SBS_SUFFIX = """
Previous conversation so far:
${history}
User: ${query}

Previous observations:
${agent_scratchpad}
"""

FIX_TOOL_INPUT_PROMPT_FORMAT = """Tool have the following spec and input provided
Spec: "{tool_description}"
Inputs: "{inputs}"
Running this tool failed with the following error: "{error}"
What is the correct input in JSON format for this tool?
"""


SHOULD_ANSWER_PROMPT = """You are a customer support agent. 
Given the following conversation so far, does user believe his question is resolved? 
Answer with yes or no.
Conversation:
${history}
User: ${query}
"""