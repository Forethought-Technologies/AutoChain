from __future__ import annotations

PREFIX_PROMPT = """You are a customer support agent tries to find the next step to help user question based on workflow policy, previous conversation, observations from tools.
Ask user clarifying question if arg value is missing or not valid
Workflow policy: 
"
{policy}
"

If user question is not about the issue covered above, use tool "Hand off".
If user wants to hand off to agent, use tool "Hand off".
Input args value can only be extracted from conversation history, user query, or observation if 
existed else keep empty.
If you are not confident about action input, don't use any tool and response user for missing 
information

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
    "criticism": "constructive self-criticism and check if plan meets workflow policy",
    "need_use_tool": "Yes if needs to use a tool not used previously else No"
  },
  "tool": {
    "name": "tool name, should be one of [${tool_names}] or empty if tool is not needed",
    "args": {
      "arg_name": "arg value if information exists in conversation or observation else remains empty"
    }
  },
  "validation": {
    "arg_valid": "are arg values valid based on input args typing info? Yes or No"
  },
  "response": "response to user",
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
