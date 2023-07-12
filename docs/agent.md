# Agent

Agent is the component that implements the interface to interact with `Chain` by deciding how to
respond to users or whether agent requires to use any tool.

It could use different prompts for different functionalities agent could have. The main goal
for agent is to plan for the next step and then either respond to user with `AgentFinish` or take a
action with `AgentAction`.

There are a few typical interactions an agent should support:

**prompt**  
Depending on agents you are building, you might want to write different agent's
planning prompts. Policy controls the steps agent should take for different situations.
Those prompts could be string templates so that later agent could substitute
different values into the prompt for different use cases

**should_answer**  
Not all the questions should be answered by agent. If agent decided that this
is not a query that should be handled by this agent, it could gracefully exit as early as
possible.

**plan**  
This is the core of the agent which takes in all the stored memory, including past
conversation history and tool outputs, which are saved to previous `AgentAction`, and prompt the
model to output either `AgentFinish` or`AgentAction` for the next step.  
`AgentFinish` means agent decides to respond back to user with a
message. While not just `plan` could output `AgentFinish`, `AgentFinish` is the **only** way to
exits the chain and wait for next user inputs.  
`AgentAction` means agent decides to use a tool and wants to perform an action before responding
to user. Once chain observes agent would like to perform an action, it will calls the
corresponding tool and store tool outputs, into the chain's memory for the next iteration of
planning.

**clarify_args_for_agent_action**  
When agent wants to take an action with tools, it is usually required to have some input arguments,
which may or may not exists in the past conversation history or action outputs. While the
smartest agent would output `AgentFinish` with response that asks user for missing information.
It might not always be the case. To decouple the problem and make is simpler for agent, we
could add another step that explicitly ask user for clarifying questions if any argument is
missing for a given tool to be used. This function will either outputs an `AgentFinish`, which
asks the clarifying question or `AgentAction` that is same as action just checked, which means
no more clarifying question is needed.  
We skipped implementing this for OpenAI agent with function calling and rely on its native
response for clarifying question.

### ConversationalAgent

This is the a basic agent with a simple default prompt template to have nice conversation with 
user. It could also use tools if provided.    
While it does not use native OpenAI function calling, this agent showcases the interaction between
memory and prompts. 
User could provide a custom prompt injected to the [prompt template](../autochain/agent/conversational_agent/prompt.py),
which contains the prompt placeholder variable.

### OpenAIFunctionAgent

At Jun 13, OpenAI released [function calling](https://platform.openai.com/docs/guides/gpt/chat-completions-api)
, which is a new way for model to use tools natively with function calling.
We introduced `OpenAIFunctionsAgent` to support native function calling when tools are provided.
To give a system message or instruction to agent via prompt, user could provide the prompt when 
creating the Agent, such as `agent = ConversationalAgent.from_llm_and_tools(llm=llm, prompt=prompt)`
