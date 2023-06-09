# Example usage

MiniChain aims to make creating a new customized agent very straight forward with as few
concepts as possible. Using MiniChain is also very simple.

## Create a conversational agent without tools

We have default `Chain` and `ConversationalAgent` implemented, so you could simply instantiate
them and start having conversation with it.

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

## Create a conversational agent with tools and customized policy

Adding tools to the agent is also similar to LangChain. In the default ConversationalAgent's
prompt. There is a placeholder for injecting customized policy that describes what agent should
do. All of the variables in the agent's prompt could be substituted with kwargs when creating
the agent.

```python
from minichain.chain.chain import Chain
from minichain.memory.buffer_memory import BufferMemory
from minichain.models.chat_openai import ChatOpenAI
from minichain.tools.base import Tool
from minichain.agent.conversational_agent.conversational_agent import ConversationalAgent

llm = ChatOpenAI(temperature=0)
tools = [Tool(
    name="Get weather",
    func=lambda *args, **kwargs: "Today is a sunny day",
    description="""This function returns the weather information"""
)]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory, tools=tools)

print(chain.run("what is the weather today")['message'])
```

## Checkout more examples in workflow_evaluation and examples

There are examples we created with mocked tools to demonstrate the ability of the agent.  
You could check out `minichain/examples` and play with agents there after setting the
OPENAI_API_KEY.  
In addition, you could run each workflow evaluation test interactively by passing `-i` flag in
the end. For example

```shell
python minichain/workflows_evaluation/order_status_request_test.py -i
```