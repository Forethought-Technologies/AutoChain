# Evaluate LangChain Agent

We created a few examples for evaluating LangChain agents with AutoChain workflow evaluation
framework. User could configure types of LangChain agent used in `LangChainWrapperChain`, which
is just a simple wrapper of LangChain to adapt to AutoChain interface.

To run any LangChain agent evaluation, user would need to install LangChain first.

```shell
pip install langchain
```

Sometimes LangChain agent would not response according to the format described in the prompt.
To walk around this problem, agent uses a custom and more lenient output parser
`CustomConvoOutputParser` and directly respond to user when output format does not match
instead of raising an exception.