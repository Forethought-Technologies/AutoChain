# AutoChain

Large language models (LLMs) have shown huge success in different text generation tasks and
enable developers to build generative agents based on natural language objectives.

However, most of the generative agents require heavy customization for a specific purpose, and 
adapting existing tools for different use cases is sometimes overwhelming. As a result, it is
still very challenging to customize on top of existing agents.

In addition, evaluating such agents powered by LLMs by trying different use
cases under different potential user scenarios is a very manual and expensive task.

AutoChain takes inspiration from LangChain and AutoGPT and aims to solve
both problems by providing a lightweight and extensible framework
for developers to build their own conversational agents using LLMs with custom tools and
[automatically evaluating](#workflow-evaluation) different user scenarios with simulated conversations.

The goal is to enable user experimentation of generative agents quickly, knowing users would
make more customizations as they are building their own agent.

## Features

- 🚀 lightweight and extensible generative agent pipeline
- 🔗 agent that can use different custom tools and support [function calling](https://platform.openai.com/docs/guides/gpt/function-calling) natively
- 💾 simple memory tracking for conversation history and tools' outputs
- 🤖 automated agent evaluation with simulated conversations

## Setup
Quick install
```shell
pip install autochain
```

Or install from source after cloning the repo

```shell
cd autochain
pyenv virtualenv 3.10.11 venv
pyenv local venv

pip install .
```

Set `PYTHONPATH` and `OPENAI_API_KEY`
```shell
export OPENAI_API_KEY=
export PYTHONPATH=`pwd`
```

Run your first conversation with agent interactively

```shell
python autochain/workflows_evaluation/order_status_request_test.py -i
```

## Example usage

If you have experience with LangChain, you already know 80% of the AutoChain interface.

AutoChain aims to make creating a new customized agent very straight forward with as few
concepts as possible. Using AutoChain is very simple.
Read more about [example usages](./docs/examples.md).

The most basic example can use our default chain and `ConversationalAgent`:

```python
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.agent.conversational_agent.conversational_agent import ConversationalAgent

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

We also added supports for native [function calling](https://platform.openai. com/docs/guides/gpt/function-calling) 
for OpenAI model. We extrapolate the function spec in OpenAI format without user explicit 
instruction, so user could follow the same `Tool` interface.   

```python
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
agent = OpenAIFunctionAgent.from_llm_and_tools(llm=llm, tools=tools)
```

Check out [more examples](./docs/examples.md) under `autochain/examples` and workflow evaluation
cases which can
also be run interactively.

## Main differences with LangChain

Although AutoChain is heavily inspired by [LangChain](https://github.com/hwchase17/langchain),
we took some design choices to make it more suitable for experimentation and iterations by
removing layers of abstraction and internal concepts.
In addition, AutoChain provides a novel way to automatically evaluate generative agents with
simulated conversations.
There are some notable differences in AutoChain's structure.

1. Flatten prompt
   In LangChain, prompts are broken into prefix, tools string, suffix and other pieces. We
   decided to have just one prompt template for each call to make prompt engineering/format
   easier and more approachable for the reader to understand how prompts are generated.
2. Up to 2 layers of abstraction
   We removed a lot of layers in LangChain and opt for a simpler interface structure, which
   comes with a cost of losing some non-essential features such as async execution and code
   duplication.
3. Simpler interaction with memory
   Passing memorized information to prompt is the most important part of this kind of framework.
   We simplify the memory interaction by having just memorized conversation and key-value pairs. All
   of them will be passed as key-value pairs and can substitute placeholders in prompts.

## Components overview

There are a few key concepts in AutoChain, which could be easily extended to build new agents.

### Chain

`Chain` is the overall *stateful* orchestrator for agent interaction. It determines when to use
tools or respond to users. `Chain` is the only stateful component, so all the interactions with
memory happen at the `Chain` level. By Default, it saves all the chat conversation history and
intermediate `AgentAction` with corresponding outputs at `prep_input` and `prep_output` steps.

`Agent` provides ways of interaction, while `Chain` determines how to
interact with agent.

Read more about the [chain concept](./docs/chain.md).

This flow diagram describes the high level picture of the default chain interaction with an agent.

![alt text](./docs/img/autochain.drawio.png)

### Agent

Agent is the *stateless* component that decides how to respond to the user or whether an agent
requires to use tools.
It could contain different prompts for different functionalities an agent could have. The main goal
for an agent is to plan for the next step, either respond to the user with `AgentFinish` or take an
action with `AgentAction`.

Read more about [agent](./docs/agent.md)

### Tool

The ability to use tools make the agent incredible more powerful as shown in LangChain and
AutoGPT. We follow a similar concept of "tool" as in LangChain here as well.
All the tools in LangChain can be easily ported over to AutoChain if you like, since they follow
a very similar interface.

Read more about [tool](./docs/tools.md).

### Memory

It is important for a chain to keep the memory for a particular conversation with a user. The memory
interface exposes two ways to save memories. One is `save_conversation` which saves the chat
history between the agent and the user, and `save_memory` to save any additional information 
for any specific business logics.

By default, memory are saved/updated in the beginning and updated in the end at `Chain` level.
Memory saves conversation history, including the latest user query, and intermediate
steps, which is a list of `AgentAction` taken with corresponding outputs.  
All memorized contents are usually provided to Agent for planning the next step.

Read more about [memory](./docs/memory.md)

## Workflow Evaluation

It is notoriously hard to evaluate generative agents in LangChain or AutoGPT. An agent's behavior
is nondeterministic and susceptible to small changes to the prompt. It can be really hard to
know if your agent is behaving correctly. The current path for evaluation is running the agent
through a large number of preset queries and evaluate the generated responses. However, that is
limited to single turn conversation, not specific to areas, and very expensive to evaluate.

To effectively evaluate agents, we introduced the workflow evaluation
which simulates the conversation between an autonomous agent and simulated users with an LLM under
different user contexts and desired outcomes of the conversation. This way, we could add test
cases for different user scenarios and use LLMs to evaluate if a conversation reached the desired
outcome.

Read more about our [evaluation strategy](./docs/workflow_evaluation.md).

### How to run workflow tests

There are two modes for running workflow tests. Interactively or running all test cases.
For example in `autochain/workflows_evaluation/refund_request_test.py`, it has already defined
a few test cases.

Running all the test cases defined in the test:

```shell
python autochain/workflows_evaluation/order_status_request_test.py
```

You can also interactively having a conversation with that agent by passing the interactive
flag `-i`:

```shell
python autochain/workflows_evaluation/order_status_request_test.py -i
```
