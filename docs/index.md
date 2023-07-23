# AutoChain

Large language models (LLMs) have shown huge success in different text generation tasks and
enable developers to build generative agents based on natural language objectives.

However, most of the generative agents require heavy customization for specific purposes, and
adapting to different use cases is sometimes overwhelming using existing tools
and framework. As a result, it is still very challenging to build a customized generative agent.

In addition, evaluating such agents powered by LLMs by trying different use
cases under different potential user scenarios is a very manual and expensive task.

AutoChain takes inspiration from LangChain and AutoGPT and aims to solve
both problems by providing a lightweight and extensible framework
for developers to build their own conversational agents using LLMs with custom tools and
[automatically evaluating](#workflow-evaluation) different user scenarios with simulated
conversations. Experienced user of LangChain would find AutoChain is easy to navigate since
they share similar but simpler concepts.

The goal is to enable quick user experiments of generative agents, knowing users would
make more customizations as they are building their own agent.

If you have any question, please feel free to reach out to Yi Lu <yi.lu@forethought.ai>

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
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i
```

## Example usage

If you have experiences with LangChain, you already know 80% of the AutoChain interfaces.

AutoChain aims to make creating a new customized agent very straight forward with as few
concepts as possible.  
Read about more [example usages](./examples.md).

The most basic example uses default chain and `ConversationalAgent`:

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

User could add a list of tools to the agent similar to LangChain

```python
tools = [Tool(
    name="Get weather",
    func=lambda *args, **kwargs: "Today is a sunny day",
    description="""This function returns the weather information"""
)]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)
print(chain.run("What is the weather today")['message'])
```

AutoChain also added supports for [function calling](https://platform.openai.
com/docs/guides/gpt/function-calling)
for OpenAI model. It extrapolates the function spec in OpenAI format without user explicit
instruction, so user could follow the same `Tool` interface.

```python
llm = ChatOpenAI(temperature=0)
agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=llm, tools=tools)
```

Check out [more examples](./examples.md) under `autochain/examples` and [workflow
evaluation](./workflow-evaluation.md) test cases which can also be run interactively.

## How does AutoChain simplify building agents?

AutoChain aims to provide a lightweight framework and simplifies the building process a few
ways comparing with other existing frameworks

1. Easy prompt update  
   Prompt engineering and iterations is one of the most important part of building generative
   agent. AutoChain makes is very obvious and easy to update prompts and visualize prompt 
   outputs. Run with `-v` flag to output verbose prompt and outputs in console.
2. Up to 2 layers of abstraction  
   Since this goal of AutoChain is enabling quick iterations, it chooses to remove most of the
   abstraction layers from alternative frameworks and make it easy to follow
3. Automated multi-turn evaluation  
   The most painful and uncertain part of building generative agent is how to evaluate its
   performance. Any change for one scenario could cause regression in other use cases. AutoChain 
   provides an easy test framework to automatically evaluate agent's ability under different 
   user scenarios.

Read more about detailed [components overview](./components_overview.md)

## Workflow Evaluation

It is notoriously hard to evaluate generative agents in LangChain or AutoGPT. An agent's behavior
is nondeterministic and susceptible to small changes to the prompt or model. It is very 
hard to know if agent is behaving correctly under different scenarios. The current path for 
evaluation is running the agent through a large number of preset queries and evaluate the 
generated responses. However, that is limited to single turn conversation, general and not 
specific to tasks and expensive to verify.

To facilitate agent evaluation, AutoChain introduces the workflow evaluation framework. This
framework runs conversations between a generative agent and LLM-simulated test users. The test
users incorporate various user contexts and desired conversation outcomes, which enables easy
addition of test cases for new user scenarios and fast evaluation. The framework leverages LLMs to
evaluate whether a given multi-turn conversation has achieved the intended outcome.

Read more about our [evaluation strategy](./workflow-evaluation.md).

### How to run workflow evaluations

There are two modes for running workflow tests. Interactively or running all test cases.
For example in `autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py`, 
there are already a few example test cases.

Running all the test cases defined in the test:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py
```

You can also have an interactive conversation with agent by passing the interactive flag `-i`:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i
```

More explanations for how AutoChain works? checkout [components overview](./components_overview.md)
