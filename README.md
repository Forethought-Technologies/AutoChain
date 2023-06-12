# MiniChain

Large language models (LLMs) have shown huge success in different text generation tasks and
enable developers to build autonomous agent based on natural language objectives.  
However, most of the agents require heavy customization for a specific purpose, and existing
tools are sometimes overwhelming to be adapted for different use cases. As the result, it is
still very challenging to customize on top of existing agents.

In addition, evaluating such autonomous agent powered by LLMs is a very manual and
expensive task by trying different use cases under different potential user scenarios.

Minichain took the inspirations from LangChain and AutoGPT and aims to solve
both problems by providing a light weighted and extensible framework
for developers to build their own conversational agent using LLMs with custom tools and
automatically evaluates different user scenarios using simulated conversations.

The goal is enable user for experimentation of generative agent quickly, knowing users would
make more customizations as they are building their own agent.

### Features

- 🚀 light weighted and extensible generative agent pipeline
- 🔗 agent that can use different custom tools
- 💾 simple memory tracks conversation history and tools outputs
- 🤖 automated agent evaluation with simulated conversations

## Setup

After cloning the repo

```shell
cd minichain
pyenv virtualenv 3.10.11 venv
pyenv local venv

pip install -r requirements.txt

export OPENAI_API_KEY=
export PYTHONPATH=`pwd`
```

Run your first conversation with agent interactively

```shell
python minichain/workflows_evaluation/order_status_request_test.py -i
```

## Example usage

If you have experience with LangChain, you already know 80% of the MiniCHain interface.
MiniChain aims to make creating a new customized agent very straight forward with as few
concepts as possible. Using MiniChain is very simple.
Read more about [example usages](./docs/examples.md)

The most basic example can use our default chain and `ConversationalAgent`

```python
from minichain.chain.chain import Chain
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.chat_openai import ChatOpenAI
from minichain.agent.conversational_agent.conversational_agent import ConversationalAgent

llm = ChatOpenAI(temperature=0)
memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm)
chain = Chain(agent=agent, memory=memory)

print(chain.run("Write me a poem about AI")['message'])
```

We could add a list of tools to the agent and chain similar to LangChain

```python
tools = [Tool(
    name="Get weather",
    func=lambda *args, **kwargs: "Today is a sunny day",
    description="""This function returns the weather information"""
)]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory, tools=tools)
```

Check out [more examples](./docs/examples.md) under `minichain/examples` and workflow evaluation
cases which can
also be run interactively.

## Main difference with LangChain

Although MiniChain is heavily inspired by [LangChain](https://github.com/hwchase17/langchain),
we took some design choices to make it more suitable for experimentation and iterations by
removing layers of abstraction and internal concepts.
In addition, MiniChain provides a novel way to automatically evaluate generative agent with 
simulated conversations.
There are some notable differences in MiniChain structure.

1. Flatten prompt
   In LangChain, prompts are broken into prefix, tools string, suffix and other pieces. We
   decided to have just one prompt template for each call to make prompt engineering/format
   easier and more approachable to reader to understand how prompts are generated.
2. Up to 2 layers of abstraction
   We removed a lot of layers in LangChain and opt for a simpler interface structure, which
   comes with a cost of losing some non-essential features such as async execution and code 
   duplication.
3. Simpler interaction with memory
   Passing memorized information to prompt is the most important part of this kind of framework.
   We simplify the memory interaction by having just memorized conversation and kv pairs. All 
   of them will be passed as kv pairs and can substitute placeholders in prompts.

## Components overview

There are a few key concepts in Minichain, which could be easily extended to build new agents.

### Chain

`Chain` is the overall orchestrator for agent interaction. It determines when to use tools or respond
to users. All the interactions with memory is limited to the `Chain` level.
`Agent` provides ways of interaction, while `Chain` determines how to
interact with agent.

Read more about the [chain concept](./docs/chain.md)

Flow diagram describes the high level picture of the default chain interaction with an agent.

![alt text](./docs/imgs/Minichain.drawio.png)

### Agent

Agent is the component that decide how to respond to user or whether agent requires to use tools.  
It could contain different prompt for different functionalities agent could have. The main goal
for agent is to plan for the next step, either respond to user with `AgentFinish` or take a
action with `AgentAction`.

Read more about [agent](./docs/agent.md)

### Tool

The ability to use tools make the agent incredible more powerful as shown in LangChain and
AutoGPT. We follow the similar concept of tool in LangChain here as well.    
All the tools in LangChain can be easily ported over to MiniChain if you like since they follow
very similar interface.

Read more about [tool](./docs/tools.md)

### Memory

It is important for chain to keep the memory for a particular conversation with user. Memory
interface expose two ways to save memories. One is `save_conversation` which saves the chat
history between agent and user, and `save_memory` to save any additional information such as
`observations` as key value pairs.  
Conversations are saved/updated before chain responds to user, and `observations` are saved
after running tools. All memorized contents are usually provided to Agent for planning
the next step.

## Workflow Evaluation

It is notoriously hard to evaluate generative agent in LangChain or AutoGPT. Agent's behavior
is nondeterministic and susceptible to small change to the prompt. It can be really hard to
know if your agent is behaving correctly. The current path for evaluation is running the agent
through a large number of preset queries and evaluate the generated responses. However, that is
limited to single turn conversation, not specific to areas, and very expensive to evaluate.

To effectively evaluate agents, we introduced the workflow evaluation
which simulate the conversation between autonomous agent and simulated users with LLM under
different user context and desired outcome of the conversation. This way, we could add test 
cases for different user scenarios and use LLM to evaluate if conversation reached the desired 
outcomes. 

Read more about our [evaluation strategy](./docs/workflow_evaluation.md)

### How to run workflow tests

There are two modes for running workflow tests. Interactively or running all test cases.
For example in `minichain/workflows_evaluation/refund_request_test.py`, it has already defined
a few test cases.
Running all the test cases defined in the test

```shell
python minichain/workflows_evaluation/order_status_request_test.py
```

You can also interactively having a conversation with that agent by passing the interactive
flag `-i`

```shell
python minichain/workflows_evaluation/order_status_request_test.py -i
```
