from __future__ import annotations

CA2_PREFIX_PROMPT = """You are a customer support agent tries to help user based on workflow policy.
Workflow policy:
{policy}
If user question is not about the issue covered above, use tool "Hand off".
If user wants to hand off to agent, use tool "Hand off".
Input args value can only be extracted from conversation history, new input, or observation if 
existed else keep empty.
If AI is not confident about action input, don't use any tool and response user for missing 
information

Assistant has access to the following tools:
"""

CODE_DESCRIPTION_PROMPT_FORMAT = """
what is the short description for the following python code
```
{code}
```
"""

SBS_FORMAT_INSTRUCTIONS = """Please respond in JSON format as described below
RESPONSE FORMAT:
{
  "thoughts": {
    "plan": "plans next step based on workflow policy and observations",
    "criticism": "constructive self-criticism and check if plan meets workflow policy",
    "need_use_tool": "Yes if needs to use a tool not used previously else No"
  },
  "tool": {
    "name": "tool name, should be one of [${tool_names}] or empty if tool is not needed",
    "args": {
      "arg_name": "arg value if information exists else remains empty"
    }
  },
  "validation": {
    "arg_valid": "are arg values valid based on input args typing info? Yes or No"
  },
  "response": "response to user",
  "workflow_finished": "Yes if reach the end of workflow else No"
}

Ensure the response can be parsed by Python json.loads"""

SBS_SUFFIX = """Previous conversation history:
${history}

New input: ${query}

${agent_scratchpad}"""



FIX_TOOL_INPUT_PROMPT_FORMAT = """Tool have the following spec and input provided
Spec: "{tool_description}"
Inputs: "{inputs}"
Running this tool failed with the following error: "{error}"
What is the correct input in JSON format for this tool?
"""