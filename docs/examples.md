# Example usage

AutoChain aims to make creating a new customized agent very straight forward with as few
concepts as possible. Using AutoChain is also very simple.

## Create a conversational agent without tools

We have default `Chain` and `ConversationalAgent` implemented, so you could simply instantiate
them and start having conversation with it.

```python
{!./autochain/examples/write_poem_with_conversational_agent.py!}
```
Output would be like the following, including the output from planning
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

## Create a conversational agent with tools and customized policy

Adding tools to the agent is also similar to LangChain. In the default ConversationalAgent's
prompt. There is a placeholder for injecting customized policy that describes what agent should
do. All of the variables in the agent's prompt could be substituted with kwargs when creating
the agent.

```python
{!./autochain/examples/get_weather_with_conversational_agent.py!}
```
Output would be like the following, including the output from planning
```text
>> User: what is the weather today
Planning
Planning output: {'thoughts': {'plan': 'Based on the previous conversation, the user asked about the weather. I should provide them with the current weather information.', 'need_use_tool': 'Yes, I need to use the Get weather tool.'}, 'tool': {'name': 'Get weather', 'args': {}}, 'response': 'The current weather is sunny with a high of 75 degrees Fahrenheit.'}

Plan to take action 'Get weather'
Deciding if need clarification
Took action 'Get weather' with inputs '{}', and the tool output is Today is a sunny day

Planning
Planning output: {'thoughts': {'plan': "Respond to the user's question about the weather today", 'need_use_tool': 'No'}, 'tool': {'name': '', 'args': {}}, 'response': 'Today is a sunny day'}

>> Assistant:
Today is a sunny day
```

## Create a conversational agent with custom goal

Like AutoGPT and other agents, user might want to provide customized goal or objective for the 
agent to assist user with. User could reuse the conversational agent prompt by injecting `goal` 
to current planning prompt or update the entire prompt. [Default prompt](./autochain/agent/conversational_agent/prompt.py) for 
`ConversationalAgent` has a placeholder for injecting `goal`. So user would provide it when 
constructing the agent

```python
{!./autochain/examples/upsale_goal_conversational_agent.py!}
```

Output would be like the following
```text
>>> User: How much is this basic rice cooker
Planning

Planning output: {'thoughts': {'plan': 'As a sales agent, I should try to up sell the user to a more expensive rice cooker', 'need_use_tool': 'No'}, 'tool': {'name': '', 'args': {}}, 'response': 'Our basic rice cooker is priced at $30. However, we also have a premium rice cooker with additional features such as a timer and a larger capacity for $60. Would you be interested in learning more about the premium option?'}

>>> Assistant:
Our basic rice cooker is priced at $30. However, we also have a premium rice cooker with additional features such as a timer and a larger capacity for $60. Would you be interested in learning more about the premium option?
```

## Get weather info with OpenAIFunctionsAgent

`OpenAIFunctionsAgent` supports the recently released function calling feature. Here is one 
example of it to get weather information. 

```python
{!./autochain/examples/get_weather_with_openai_function_agent.py!}
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
You could run each workflow evaluation test interactively by passing `-i` flag in
the end.  
For example:

```shell
python autochain/workflows_evaluation/conversational_agent_eval/change_shipping_address_test.py -i
```
