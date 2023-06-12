# Agent

Agent is the component that implement the interface to interact with `Chain` by deciding how to
respond to user or whether agent requires to use tools.

It could contain different prompt for different functionalities agent could have. The main goal
for agent is to plan for the next step, either respond to user with `AgentFinish` or take a
action with `AgentAction`.

There are a few typical interactions an agent should support:

**prompt** Depending on agents you are building, you might want to write different agent's
planning prompt. policy controls the steps agent should take for different situation.
Those prompt could be string template so that later one the same agent could substitute
different values into the prompt for different use cases

**should_answer**: not all the question should be answered by agent. If agent decides that this
is not a query that should be handled by this agent, it could gracefully exits as early as
possible.

**plan**: This is the core of the agent which takes in all the stored memory, including past
conversation history and tool output, named `observations`, and prompt the model to output
either `AgentFinish` or`AgentAction`.  
`AgentFinish` means agent decide to respond back to user with a
message. While not just `plan` could output `AgentFinish`, `AgentFinish` is the **only** way to
exits the chain and wait for next user inputs.  
`AgentAction` means agent decide to use a tool and wants to perform an action before responding
to user. Once chain observe agent would like to perform an action, it will calls the
corresponding tool and store tool outputs, named `observations`, into the chain's memory for
future interactions. At this point, there is no message respond back to user.

**clarify_args_for_agent_action**
When agent wants to take an action with tools, it usually requires to have some input arguments,
which may or may not exists in the past conversation history or observations. While the
smartest agent would output `AgentFinish` with response that asks user for missing information.
It might not always be the case. To decouple the problem and make is simpler for agent, we
could add another step that explicitly ask user for clarifying questions when any argument is
missing for a given tool to be used. This function will either outputs an `AgentFinish`, which
asks the clarifying question or `AgentAction` that is same as action just checked, which means
no more clarifying question is needed.

## Differences with LangChain

As we design the agent, we aims to make it easier to understand and troubleshoot as we believe
MiniChain is a framework for experimentation. So we tool some design choices to remove layers
of abstractions/concepts. Some notable differences are

1. Different way to create and update prompt. In LangChain, prompts are broken into prefix,
   tools string, suffix and other pieces. While that is efficient to reuse, it can also be hard
   to understand what prompt is used and how to update it. So in MiniChain, there is just one
   prompt with placeholders that can be substituted with variables.
2. Fewer layers of abstraction. Although inheritance is commonly used, we opt to have just 2
   layers of abstractions, `BaseAgent` and the actual agent implementation. However, this comes 
   with a cost of not able to share as much code across agents. This is not very concerning to us 
   because MiniChain aims to enable quick experimentation.