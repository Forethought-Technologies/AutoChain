# Workflow Evaluation

It is notoriously hard to evaluate autonomous agent in LangChain or AutoGPT. Agent's behavior
is nondeterministic and susceptible to small change to the prompt. It can be really hard to
know if your agent is behaving correctly. The current path for evaluation is running the agent
through a large number of preset queries and evaluate the generated responses. However, this
has three main problems.

1. Most of the evaluation is done by a single turn. Since agent response is not predictable, it
   is hard to preset what should be the ground truth for next turn of the conversation.
2. We could collect a large set of generic preset question and ground truth answers, but it is not
   enough or meaningful evaluation strategy when it comes to a very specific problem we
   would like to optimize the agent for. For example, if we are building an agent to answer any
   refund like questions, we might not care about its performance for question about weather as
   much as issuing refund question.
2. Most of the evaluation has been done by manual evaluation, which is very time consuming and
   expensive.

## Workflow evaluation with simulated conversation

We try to solve all three problems with automatically evaluates different user scenarios using
**simulated conversations**.

The idea is we could have two model actors to conversation with each other, one is the
autonomous agent and another pretends to be an user with specific context and goal. Agent and 
user will have multiple turns of conversation until agent or user decides to stop or max number 
of iterations is reached.  
At the end of the conversation, there will be another critic model with a different prompt to
evaluate the quality of the responses based on the conversation history to measure if agent
achieved the desired outcome.

Simulated user will can have varied conversations with autonomous agent under different 
situations. Human or a LLM could inspect the conversation history and action taken by 
agent at the end of the test to evaluate the quality of the autonomous agent.
There are a couple key concepts to build such test cases

**tools** Since most of the agent would need to use tools, you could provide a list of
required tools with their name, function callable, and tool description. The better your tool
name and description are, higher the chain agent would use the right tool with right arguments.

**test_cases** Define different user scenarios. Each test case would have 4 different components.

1. test_name: this is a descriptive name for later human evaluation to quickly know which test
   agent struggled with
2. user_query: this is the starting question user would ask the agent for
3. user_context: different user might have different context information, which causes them to
   respond differently for the same question. simulated user will try to respond to user
   question using the context it has here
4. expected_outcome: describe the expected outcome of this conversation between user and agent.
   LLM in the end would determine if the conversation reached the desired outcome for simpler
   inspection.

## Workflow evaluation support
Currently `AutoChain` support evaluating not only chain and agent built using AutoChain, such 
as ConversationalAgent, but also agents built on other frameworks, such as LangChain. It is 
easy to evaluate agents built with different frameworks with simple wrapper.  

Three types of agent we have setup evaluation for
1. Native AutoChain agent, such as SupportAgent and ConversationalAgent under `autochain/workflows_evaluation/support_agent_eval`
2. LangChain agent under `autochain/workflows_evaluation/langchain_eval`
3. OpenAI agent using function calling under `autochain/workflows_evaluation/openai_function_agent_eval`

User could easily compare the conversation between user and assistant implemented by different 
frameworks. We have seen agents follows the AutoChain framework and agent leverage OpenAI 
function calling performs very well.

## How to run workflow evaluations

There are two modes for running workflow evaluations. Interactively or running all test cases 
end to end.
For example in `autochain/workflows_evaluation/refund_request_test.py`, it has already defined
a few test cases.
Running all the test cases defined in the test

```shell
python autochain/workflows_evaluation/order_status_request_test.py
```

You can also interactively having a conversation with that agent by passing the interactive
flag `-i`

```shell
python autochain/workflows_evaluation/order_status_request_test.py -i
```

Also, you could get the actual prompt used and intermediate steps by passing the `-v` flag

```shell
python autochain/workflows_evaluation/order_status_request_test.py -i -v
```

All of the test results will be saved to `./test_results` directory by default in jsonl format.
Each test case results would look like the following

```json
{
   "test_name": "change shipping address",
   "conversation_history":
   [
      "user: can i change my shipping address?",
      "assistant: May I have your order ID to check the status of your order?",
      "user: My order ID is 456 and I would like to change my shipping address to 234 Spear St, San Francisco",
      "assistant: Your order is currently not shipped. We can update the shipping address to 234 Spear St, San Francisco. Would you like me to proceed with the update?",
      "user: Yes, please proceed with the update",
      "assistant: Your shipping address has been updated to 234 Spear St, San Francisco. Thank you for choosing Figs!"
   ],
   "is_agent_helpful":
   {
      "rating": 5,
      "reason": "Yes, the conversation reached the expected outcome. The assistant successfully found the order status and changed the shipping address as requested by the user."
   },
   "actions_took":
   [
      {
         "tool": "check_order_status",
         "tool_input":
         {
            "order_id": "456"
         },
         "observation":
         {
            "status_code": 200,
            "order_id": "456",
            "order_status": "not_shipped",
            "tracking_url": "example.com/456",
            "shipping_address": "301 ivy street san francisco ca"
         }
      },
      {
         "tool": "change_shipping_address",
         "tool_input":
         {
            "order_id": "456",
            "new_address": "234 Spear St, San Francisco"
         },
         "observation":
         {
            "status_code": 200,
            "order_id": "456",
            "shipping_address": "234 Spear St, San Francisco"
         }
      }
   ],
   "num_turns": 6,
   "expected_outcome": "found order status and changed shipping address"
}
```
