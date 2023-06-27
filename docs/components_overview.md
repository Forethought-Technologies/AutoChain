# Components overview

There are a few key concepts in AutoChain, which could be easily extended to build new agents.

### Chain

`Chain` is the overall *stateful* orchestrator for agent interaction. It determines when to use
tools or respond to users. `Chain` is the only stateful component, so all the interactions with
memory happen at the `Chain` level. By Default, it saves all the chat conversation history and
intermediate `AgentAction` with corresponding outputs at `prep_input` and `prep_output` steps.

`Agent` provides ways of interaction, while `Chain` determines how to
interact with agent.

Read more about the [chain concept](./chain.md).

### Agent

Agent is the *stateless* component that decides how to respond to the user or whether an agent
requires to use tools.
It could contain different prompts for different functionalities an agent could have. The main goal
for an agent is to plan for the next step, either respond to the user with `AgentFinish` or take an
action with `AgentAction`.

Read more about [agent](./agent.md).

### Tool

The ability to use tools make the agent incredible more powerful as shown in LangChain and
AutoGPT. We follow a similar concept of "tool" as in LangChain here as well.
All the tools in LangChain can be easily ported over to AutoChain if you like, since they follow
a very similar interface.

Read more about [tool](./tool.md).

### Memory

It is important for a chain to keep the memory for a particular conversation with a user. The
memory
interface exposes two ways to save memories. One is `save_conversation` which saves the chat
history between the agent and the user, and `save_memory` to save any additional information
for any specific business logics.

By default, memory are saved/updated in the beginning and updated in the end at `Chain` level.
Memory saves conversation history, including the latest user query, and intermediate
steps, which is a list of `AgentAction` taken with corresponding outputs.  
All memorized contents are usually provided to Agent for planning the next step.

Read more about [memory](./memory.md)