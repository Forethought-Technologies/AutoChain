# Example usage

AutoChain aims to make creating a new customized agent very straight forward with as few
concepts as possible. Using AutoChain is also very simple.

## Create a conversational agent without tools

We have default `Chain` and `ConversationalAgent` implemented, so you could simply instantiate
them and start having conversation with it.

```python
{!./docs_src/tutorial/examples/tutorial001.py!}
```
Output would be like the following, including the output from planning
```text
Planning
Full output: {'thoughts': {'plan': 'Based on the previous conversation, it seems like the user is interested in AI. I could suggest some resources or articles about AI poetry.', 'need_use_tool': 'No'}, 'tool': {}, 'response': "Sure, I'd be happy to help! Here are a few resources you might find interesting:\n- 'The Poet and the Machine' by Adam Roberts\n- 'The AI-Aided Art of Writing Poetry' by Janelle Shane\n- 'The Poet of the Future: AI and the Evolution of Creativity' by David C. Stolinsky\nI hope you find these helpful!"}

>>> Assistant:
Sure, I'd be happy to help! Here are a few resources you might find interesting:
- 'The Poet and the Machine' by Adam Roberts
- 'The AI-Aided Art of Writing Poetry' by Janelle Shane
- 'The Poet of the Future: AI and the Evolution of Creativity' by David C. Stolinsky
I hope you find these helpful!
```

## Create a conversational agent with tools and customized policy

Adding tools to the agent is also similar to LangChain. In the default ConversationalAgent's
prompt. There is a placeholder for injecting customized policy that describes what agent should
do. All of the variables in the agent's prompt could be substituted with kwargs when creating
the agent.

```python
{!./docs_src/tutorial/examples/tutorial002.py!}
```
Output would be like the following, including the output from planning
```text
Planning
Full output: {'thoughts': {'plan': 'Based on the previous conversation, the user asked about the weather. I should provide them with the current weather information.', 'need_use_tool': 'Yes, I need to use the Get weather tool.'}, 'tool': {'name': 'Get weather', 'args': {}}, 'response': 'The current weather is sunny with a high of 75 degrees Fahrenheit.'}

Plan to take action 'Get weather'
Deciding if need clarification
Took action 'Get weather' with inputs '{}', and the observation is Today is a sunny day

Planning
Full output: {'thoughts': {'plan': "Respond to the user's question about the weather today", 'need_use_tool': 'No'}, 'tool': {'name': '', 'args': {}}, 'response': 'Today is a sunny day'}

>>> Assistant:
Today is a sunny day
```

## Checkout more examples in workflow_evaluation and examples

There are examples we created with mocked tools to demonstrate the ability of the agent.
You could check out `autochain/examples` and play with agents there after setting the
OPENAI_API_KEY.
In addition, you could run each workflow evaluation test interactively by passing `-i` flag in
the end. For example

```shell
python autochain/workflows_evaluation/order_status_request_test.py -i
```
