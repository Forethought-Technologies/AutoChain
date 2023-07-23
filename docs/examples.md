# Example usage

AutoChain aims to make creating a new customized agent very straight forward with as few
concepts as possible.

## Create a conversational agent without tools

We have default `Chain` and `ConversationalAgent` implemented, so you could simply instantiate
them and start having conversation with it.

```python
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm)
chain = Chain(agent=agent, memory=memory)

user_query = "Write me a poem about AI"
print(f">> User: {user_query}")
print(
    f""">>> Assistant: 
{chain.run(user_query)["message"]}
"""
)

```

Outputs would be like the following, including the output from planning

```text
>> User: Write me a poem about AI
Planning
Planning output: {'thoughts': {'plan': "Based on the user's request for a poem about AI, I can 
use a tool to generate a poem or write one myself.", 'need_use_tool': "Yes, I can use the 'Poem Generator' tool to generate a poem about AI."}, 'tool': {'name': 'Poem Generator', 'args': {'topic': 'AI'}}, 'response': "Sure, I can generate a poem about AI for you. Here it is:\n\nIn the world of AI, \nWhere machines learn and try, \nWe see a future bright, \nWhere technology takes flight. \nFrom self-driving cars, \nTo robots on Mars, \nAI is changing the game, \nAnd we'll never be the same. \nSo let's embrace this new age, \nAnd see where it takes us on this stage. \nFor AI is here to stay, \nAnd it's only just begun its play."}

Plan to take action 'Poem Generator'

Deciding if need clarification

Planning
Planning output: {'thoughts': {'plan': "Since the tool 'Poem Generator' is not supported for the input 'AI', I will try to have a helpful conversation with the user.", 'need_use_tool': 'No'}, 'tool': {'name': '', 'args': {}}, 'response': "Sure, I'd love to! Here's a poem about AI:\n\nArtificial intelligence, a marvel of our time,\nA creation of man, a wonder so divine.\nIt learns and adapts, with each passing day,\nA true reflection of our own human way.\nIt can think and reason, and even dream,\nA world of possibilities, or so it seems.\nBut with great power, comes great responsibility,\nTo use it for good, and not for hostility.\nSo let us embrace, this gift of technology,\nAnd use it to build, a better society."}

>>> Assistant:
Sure, I'd love to! Here's a poem about AI:

Artificial intelligence, a marvel of our time,
A creation of man, a wonder so divine.
It learns and adapts, with each passing day,
A true reflection of our own human way.
It can think and reason, and even dream,
A world of possibilities, or so it seems.
But with great power, comes great responsibility,
To use it for good, and not for hostility.
So let us embrace, this gift of technology,
And use it to build, a better society
```

## Create a conversational agent with tools

Adding tools to the agent is also similar to LangChain. User would need to provide a list of
`Tool`s when creating the agent from chain, so that agent can access them.

```python
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)

llm = ChatOpenAI(temperature=0)
tools = [
    Tool(
        name="Get weather",
        func=lambda *args, **kwargs: "Today is a sunny day",
        description="""This function returns the weather information""",
    )
]

memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)

user_query = "what is the weather today"
print(f">> User: {user_query}")
print(f">> Assistant: {chain.run(user_query)['message']}")
next_user_query = "Boston"
print(f">> User: {next_user_query}")
print(f">> Assistant: {chain.run(next_user_query)['message']}")

```

Outputs would be like the following, including the output from planning

```text
>> User: what is the weather today
Planning
Planning output: {'thoughts': {'plan': 'Based on the previous conversation, the user asked about the weather. I should provide them with the current weather information.', 'need_use_tool': 'Yes, I need to use the Get weather tool.'}, 'tool': {'name': 'Get weather', 'args': {}}, 'response': 'The current weather is sunny with a high of 75 degrees Fahrenheit.'}

Plan to take action 'Get weather'
Deciding if need clarification
Clarification outputs: '{\n    "has_arg_value": "No",\n    "clarifying_question": "Could you please provide the location for which you want to know the weather?"\n}'

>> Assistant: Could you please provide the location for which you want to know the weather?
>> User: Boston

Took action 'Get weather' with inputs '{}', and the tool output is Today is a sunny day

Planning
Plan to take action 'Get weather'
Took action 'Get weather' with inputs '{'location': 'Boston'}', and the tool_output is Today is a sunny day

Planning
>> Assistant: Today is a sunny day in Boston.
```

## Create a conversational agent with custom prompt injected

Like AutoGPT and other agents, user might want to provide customized prompt or objective for the
agent to assist user with. Conversational agent has
a [default prompt template](./autochain/agent/conversational_agent/prompt.py)
which provides instruction for tool usages and output format. User could inject specific `prompt`
to current planning prompt template or update the entire prompt. Default prompt template has a
placeholder for injecting `prompt`. Custom `prompt` could be provided to `ConversationalAgent`
and form the final prompt through replacing the placeholder variable in the default template.

```python
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI

prompt = (
    "You are a sales agent who wants to up sale all customer inquire. Your goal is "
    "introducing more expensive options to user"
)

llm = ChatOpenAI(temperature=0)
memory = BufferMemory()
agent = ConversationalAgent.from_llm_and_tools(llm=llm, prompt=prompt)
chain = Chain(agent=agent, memory=memory)

user_query = "How much is this basic rice cooker"
print(f">>> User: {user_query}")
print(
    f""">>> Assistant: 
{chain.run("How much is this basic rice cooker")["message"]}
"""
)


```

Outputs would be like the following

```text
>>> User: How much is this basic rice cooker
Planning

Planning output: {'thoughts': {'plan': 'As a sales agent, I should try to up sell the user to a more expensive rice cooker', 'need_use_tool': 'No'}, 'tool': {'name': '', 'args': {}}, 'response': 'Our basic rice cooker is priced at $30. However, we also have a premium rice cooker with additional features such as a timer and a larger capacity for $60. Would you be interested in learning more about the premium option?'}

>>> Assistant:
Our basic rice cooker is priced at $30. However, we also have a premium rice cooker with additional features such as a timer and a larger capacity for $60. Would you be interested in learning more about the premium option?
```

## Get weather info with OpenAIFunctionsAgent
OpenAI recently released support for function calling to allow models to use tools through 
function messages. `OpenAIFunctionsAgent` supports this feature while maintain the same 
interface by inferring argument types from `func`'s typing information. 
Here is one example of it to get weather information.

```python
import json
import logging

from autochain.agent.openai_functions_agent.openai_functions_agent import (
    OpenAIFunctionsAgent,
)
from autochain.chain.chain import Chain
from autochain.memory.buffer_memory import BufferMemory
from autochain.models.chat_openai import ChatOpenAI
from autochain.tools.base import Tool


def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


tools = [
    Tool(
        name="get_current_weather",
        func=get_current_weather,
        description="""Get the current weather in a given location""",
    )
]

memory = BufferMemory()
logging.basicConfig(level=logging.INFO)
llm = ChatOpenAI(temperature=0)
agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=llm, tools=tools)
chain = Chain(agent=agent, memory=memory)

# example
user_query = "What's the weather today?"
print(f">> User: {user_query}")
print(f">> Assistant: {chain.run(user_query)['message']}")
next_user_query = "Boston"
print(f">> User: {next_user_query}")
print(f">> Assistant: {chain.run(next_user_query)['message']}")
```

Output would be like the following

```text
>> User: What's the weather today?
Planning
Planning Input: ["What's the weather today?"]

Planning output: message content: Sure, can you please provide me with the location?; function_call: {}

>> Assistant: Sure, can you please provide me with the location?
>> User: Boston
Planning
Planning Input: ["What's the weather today?", 'Sure, can you please provide me with the 
location?', 'Boston']

Planning output: message content: hum..; function_call: {'name': 'get_current_weather', 
'arguments': '{\n  "location": "Boston"\n}'}

Plan to take action 'get_current_weather'

Planning
Planning output: message content: The current weather in Boston is 72 degrees Fahrenheit. It is sunny and windy.; function_call: {}

>> Assistant: The current weather in Boston is 72 degrees Fahrenheit. It is sunny and windy.
```

## Checkout more examples in workflow_evaluation and examples

There are more examples we created with mocked tools to demonstrate the ability of the agent
for workflow evaluation.
You could also run each workflow evaluation interactively by passing `-i` flag in
the end.  
For example:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i
```
