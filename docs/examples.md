# Example usage

AutoChain aims to make creating a new customized agent very straight forward with as few
concepts as possible. Using AutoChain is also very simple.

## Create a conversational agent without tools

We have default `Chain` and `ConversationalAgent` implemented, so you could simply instantiate
them and start having conversation with it.

```python
{!./docs_src/tutorial/examples/tutorial001.py!}
```

## Create a conversational agent with tools and customized policy

Adding tools to the agent is also similar to LangChain. In the default ConversationalAgent's
prompt. There is a placeholder for injecting customized policy that describes what agent should
do. All of the variables in the agent's prompt could be substituted with kwargs when creating
the agent.

```python
{!./docs_src/tutorial/examples/tutorial002.py!}
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
