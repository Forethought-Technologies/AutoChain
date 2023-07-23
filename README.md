# AutoChain

Large language models (LLMs) have shown huge success in different text generation tasks and
enable developers to build generative agents based on objectives expressed in natural language.

However, most generative agents require heavy customization for specific purposes, and
supporting different use cases can sometimes be overwhelming using existing tools
and frameworks. As a result, it is still very challenging to build a custom generative agent.

In addition, evaluating such generative agents, which is usually done by manually trying different
scenarios, is a very manual, repetitive, and expensive task.

AutoChain takes inspiration from LangChain and AutoGPT and aims to solve
both problems by providing a lightweight and extensible framework
for developers to build their own agents using LLMs with custom tools and
[automatically evaluating](#workflow-evaluation) different user scenarios with simulated
conversations. Experienced user of LangChain would find AutoChain is easy to navigate since
they share similar but simpler concepts.

The goal is to enable rapid iteration on generative agents, both by simplifying agent customization
and evaluation.

If you have any questions, please feel free to reach out to Yi Lu <yi.lu@forethought.ai>

## Features

- 🚀 lightweight and extensible generative agent pipeline.
- 🔗 agent that can use different custom tools and
  support OpenAI [function calling](https://platform.openai.com/docs/guides/gpt/function-calling)
- 💾 simple memory tracking for conversation history and tools' outputs
- 🤖 automated agent multi-turn conversation evaluation with simulated conversations

## Setup

Quick install

```shell
pip install autochain
```

Or install from source after cloning this repository

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
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i
```

## How does AutoChain simplify building agents?

AutoChain aims to provide a lightweight framework and simplifies the agent building process in a
few
ways, as compared to existing frameworks

1. Easy prompt update  
   Engineering and iterating over prompts is a crucial part of building generative
   agent. AutoChain makes it very easy to update prompts and visualize prompt
   outputs. Run with `-v` flag to output verbose prompt and outputs in console.
2. Up to 2 layers of abstraction  
   As part of enabling rapid iteration, AutoChain chooses to remove most of the
   abstraction layers from alternative frameworks
3. Automated multi-turn evaluation  
   Evaluation is the most painful and undefined part of building generative agents. Updating the
   agent to better perform in one scenario often causes regression in other use cases. AutoChain
   provides a testing framework to automatically evaluate agent's ability under different
   user scenarios.

## Example usage

If you have experience with LangChain, you already know 80% of the AutoChain interfaces.

AutoChain aims to make building custom generative agents as straightforward as possible, with as
little abstractions as possible.

The most basic example uses the default chain and `ConversationalAgent`:

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

Just like in LangChain, you can add a list of tools to the agent

```python
tools = [
    Tool(
        name="Get weather",
        func=lambda *args, **kwargs: "Today is a sunny day",
        description="""This function returns the weather information"""
    )
]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)
print(chain.run("What is the weather today")['message'])
```

AutoChain also added support
for [function calling](https://platform.openai.com/docs/guides/gpt/function-calling)
in OpenAI models. Behind the scenes, it turns the function spec into OpenAI format without explicit
instruction, so you can keep following the same `Tool` interface you are familiar with.

```python
llm = ChatOpenAI(temperature=0)
agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=llm, tools=tools)
```

See [more examples](./docs/examples.md) under `autochain/examples` and [workflow
evaluation](./docs/workflow-evaluation.md) test cases which can also be run interactively.

Read more about detailed [components overview](./docs/components_overview.md)

## Workflow Evaluation

It is notoriously hard to evaluate generative agents in LangChain or AutoGPT. An agent's behavior
is nondeterministic and susceptible to small changes to the prompt or model. As such, it is
hard to know what effects an update to the agent will have on all relevant use cases.

The current path for
evaluation is running the agent through a large number of preset queries and evaluate the
generated responses. However, that is limited to single turn conversation, general and not
specific to tasks and expensive to verify.

To facilitate agent evaluation, AutoChain introduces the workflow evaluation framework. This
framework runs conversations between a generative agent and LLM-simulated test users. The test
users incorporate various user contexts and desired conversation outcomes, which enables easy
addition of test cases for new user scenarios and fast evaluation. The framework leverages LLMs to
evaluate whether a given multi-turn conversation has achieved the intended outcome.

Read more about our [evaluation strategy](./docs/workflow-evaluation.md).

### How to run workflow evaluations

You can either run your tests in interactive mode, or run the full suite of test cases at once.
`autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py` contains a few
example test cases.

To run all the cases defined in a test file:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py
```

To run your tests interactively `-i`:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i
```

Looking for more details on how AutoChain works? See
our [components overview](./docs/components_overview.md)
