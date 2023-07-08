# AutoChain

Large language models (LLMs) have shown huge success in different text generation tasks and
enable developers to build generative agents based on natural language objectives.

However, most of the generative agents require heavy customization for a specific purpose, and
adapting to different use cases is sometimes overwhelming using existing tools
and framework. As a result, it is still very challenging to build a customized generative agent.

In addition, evaluating such agents powered by LLMs by trying different use
cases under different potential user scenarios is a very manual and expensive task.

AutoChain takes inspiration from LangChain and AutoGPT and aims to solve
both problems by providing a lightweight and extensible framework
for developers to build their own conversational agents using LLMs with custom tools and
[automatically evaluating](#workflow-evaluation) different user scenarios with simulated
conversations. So experiences user of LangChain would find AutoChain is easy to navigate since
they share similar concepts.

The goal is to enable user experimentation of generative agents quickly, knowing users would
make more customizations as they are building their own agent.

If you have any question, please feel free to reach out to Yi Lu <yi.lu@forethought.ai>

## Features

- 🚀 lightweight and extensible generative agent pipeline made easy to LangChain users.
- 🔗 agent that can use different custom tools and
  support [function calling](https://platform.openai.com/docs/guides/gpt/function-calling)
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
python autochain/workflows_evaluation/conversational_agent_eval/change_shipping_address_test.py -i
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

AutoChain also added supports for [function calling](https://platform.openai.
com/docs/guides/gpt/function-calling)
for OpenAI model. It extrapolates the function spec in OpenAI format without user explicit
instruction, so user could follow the same `Tool` interface.

```python
llm = ChatOpenAI(temperature=0)
agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=llm, tools=tools)
```

Check out [more examples](./docs/examples.md) under `autochain/examples` and [workflow
evaluation](./docs/workflow-evaluation.md) test cases which can also be run interactively.

## How does AutoChain simplify building agents?

AutoChain aims to provide a lightweight framework and simplifies the building process a few
ways comparing with other existing frameworks

1. Visible prompt used
   Prompt engineering and iterations is one of the most important part of building generative
   agent. AutoChain makes is very obvious and easy to update prompts.
2. Up to 2 layers of abstraction
   Since this goal of AutoChain is enabling quick iterations, it chooses to remove most of the
   abstraction layers from alternative framework and make it easy to follow
3. Automated multi-turn evaluation
   The most painful and uncertain part of building generative agent is how to evaluate its
   performance. Any change could cause regression in other use cases. AutoChain provides an
   easy test framework to automatically evaluate agent's ability under different user scenarios.

Read mode about detailed [components overview](./docs/components_overview.md)

## Workflow Evaluation

It is notoriously hard to evaluate generative agents in LangChain or AutoGPT. An agent's behavior
is nondeterministic and susceptible to small changes to the prompt. It can be really hard to
know if your agent is behaving correctly. The current path for evaluation is running the agent
through a large number of preset queries and evaluate the generated responses. However, that is
limited to single turn conversation, not specific to areas, and very expensive to evaluate.

To effectively evaluate agents, AutoChain introduced the workflow evaluation
which simulates the conversation between an generative agent and simulated users with an LLM under
different user contexts and desired outcomes of the conversation. This way, we could add test
cases for different user scenarios and use LLMs to evaluate if a conversation reached the desired
outcome.

Read more about our [evaluation strategy](./docs/workflow-evaluation.md).

### How to run workflow tests

There are two modes for running workflow tests. Interactively or running all test cases.
For example in `autochain/workflows_evaluation/conversational_agent_eval
/change_shipping_address_test.py`, it has already defined a few test cases.

Running all the test cases defined in the test:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/change_shipping_address_test.py
```

You can also have an interactive conversation with agent by passing the interactive flag `-i`:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/change_shipping_address_test.py -i
```

More explanations for how AutoChain works? checkout [components overview](./docs/components_overview.md)

