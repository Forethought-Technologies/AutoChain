# Agent

Agent is the component that implements the interface to interact with `Chain` by deciding how to
respond to users or whether agent requires to use any tool.

It could use different prompts for different functionalities agent could have. The main goal
for agent is to plan for the next step and then either respond to user with `AgentFinish` or take a
action with `AgentAction`.

There are a few typical interactions an agent should support:

**prompt** Depending on agents you are building, you might want to write different agent's
planning prompts. Policy controls the steps agent should take for different situations.
Those prompts could be string templates so that later agent could substitute
different values into the prompt for different use cases

**should_answer**: not all the questions should be answered by agent. If agent decided that this
is not a query that should be handled by this agent, it could gracefully exit as early as
possible.

**plan**: This is the core of the agent which takes in all the stored memory, including past
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

## Differences with LangChain

As we design the agent, we aim to make it easier to understand and troubleshoot as we believe
AutoChain is a framework for experimentation. So we took some design choices to remove layers
of abstractions/concepts. Some notable differences are

1. Different way to create and update prompt. In LangChain, prompts are broken into prefix,
   tools string, suffix and other pieces. While that is efficient for reuse, it can also be hard
   to understand what prompt is used and how to update it. So in AutoChain, there is just one
   prompt with placeholders that can be substituted with variables from inputs.
2. Fewer layers of abstraction. Although inheritance is commonly used, we opt to have just 2
   layers of abstractions, `BaseAgent` and the actual agent implementation for example. However,
   this comes with a cost of not able to share as much code across agents. This is not very
   concerning to us because AutoChain aims to just enable quick experimentation.
3. Allows agent to follow different prompts at each step. We also revisit the prompt used in
   LangChain and optimize them to allow agent to have smoother conversations.

From our workflow evaluation examples, we have observed that agent follow `AutoChain` framework
performs better than LangChain zero shot agents in term of quality of response and ability to
use tools.

## Supported Agent Types

We have added several different types of agent to showcase how to add new agents to AutoChain.

### ConversationalAgent

This is the a basic agent with a simple and fixed prompt to have nice conversation with user.
It could also use tools if provided.    
While it does not use native OpenAI function calling, this agent showcases the interaction between
memory and prompts. Also it supports using ChatGPT model before `0613`.

### SupportAgent

`SupportAgent` is an enhanced version of `ConversationalAgent` which has a policy in mind when
having the conversation with user. It will try to follow the policy as much as possible and
gracefully handoff when it is not sure.

### OpenAIFunctionAgent

At Jun 13, OpenAI released [function calling](https://platform.openai.
com/docs/guides/gpt/chat-completions-api)
, which is a new way for model to use tools natively with function calling.
We introduced `OpenAIFunctionAgent` to support native function calling when tools are provided.
