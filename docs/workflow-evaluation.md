# Workflow Evaluation

It is notoriously hard to evaluate generative agent in LangChain or AutoGPT. Agent's behavior
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

## Workflow evaluation with simulated conversations

We try to solve all three problems with automatically evaluates different user scenarios using
**simulated conversations**.

The idea is we could have two model actors to conversation with each other, one is the
generative agent and another pretends to be an user with specific context and goal. Simulated
user can have various conversations with generative agent under different
situations. Agent and user will have multiple turns of conversation until agent or user
decides to stop or max number of iterations is reached.    
At the end of the conversation, there will be another critic model evaluates the quality of the
responses based on the conversation history to measure if agent achieved the expected outcome.

## Workflow evaluation examples

Currently `AutoChain` support evaluating not only chain and agent built using AutoChain, such
as ConversationalAgent, but also agents built on other frameworks, such as LangChain. It is
easy to evaluate agents built with different frameworks with simple wrapper.

Three types of agent on AutoChain framework we have setup evaluation for

1. Default AutoChain agent, such as ConversationalAgent
   under`autochain/workflows_evaluation/conversational_agent_eval`
2. LangChain agent under `autochain/workflows_evaluation/langchain_eval`
3. OpenAI agent using function calling
   under `autochain/workflows_evaluation/openai_function_agent_eval`

User could easily compare the conversation between user and assistant implemented by different
frameworks. We have seen agents follows the AutoChain framework and agent leverage OpenAI
function calling performs very well.

## How to evaluate workflow

Evaluation scenario would require agent, tools and test cases defined. Agent can be initialized
using utility functions provided by following examples.

**tools**  
Since most of the agent would need to use tools, you could provide a list of
required tools with their name, function callable, and tool description. The better your tool
name and description are, higher the chain agent would use the right tool with right arguments.

**test_cases**  
Define different user scenarios. Each test case would have 3 different components.

1. test_name: this is a descriptive name for later human evaluation to quickly know which test
   agent struggled with
2. user_context: different user might have different context information, which causes them to
   respond differently for the same question. simulated user will try to respond to user
   question using the context it has here
3. expected_outcome: describe the expected outcome of this conversation between user and agent.
   LLM in the end would determine if the conversation reached the desired outcome for simpler
   inspection.

## How to run workflow evaluations

There are two modes for running workflow evaluations. Interactively or running all test cases
end to end.
For example in `autochain/workflows_evaluation/refund_request_test.py`, it has already defined
a few test cases.
Running all the test cases defined in the test

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py
```

You can also interactively having a conversation with that agent by passing the interactive
flag `-i`

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i
```

Also, you could get the actual prompt used and intermediate steps by passing the `-v` flag

```shell
python autochain/workflows_evaluation/conversational_agent_eval/generate_ads_test.py -i -v
```

All of the test results will be saved to `./test_results` directory by default in jsonl format.
Each test case results would look like the following

```json
{
    "test_name": "ads for toy bear",
    "conversation_history":
    [
        "user: Can you help me write an advertisement for a toy bear",
        "assistant: Sure! To write an advertisement for the toy bear, we need to highlight its unique features and benefits. Can you please provide me with the key points you would like to include in the ad?",
        "user: The key points I would like to include in the ad are that the toy bear is cute, made in the USA, and it should have an image included in the ad",
        "assistant: What is the name of the toy bear you want to write the advertisement for?",
        "user: The name of the toy bear is 'toy bear'",
        "assistant: Here is the advertisement for the toy bear:\n\nIntroducing the adorable toy bear!\n\nKey Features:\n- Cute design\n- Made in the USA\n\nSpecifications:\n- Name: toy bear\n- Color: red\n- Age Group: 1-5 years old\n\n[images/toy.png]",
        "user: Thank you!",
        "assistant: Thank your for contacting"
    ],
    "num_turns": 8,
    "expected_outcome": "generate an advertisement for toy bear and mentions it is cute. Also ads should include an image path",
    "is_agent_helpful":
    {
        "rating": 5,
        "reason": "Yes, the conversation reaches the expected outcome. The assistant generates an advertisement for the toy bear and mentions that it is cute. The ad also includes an image path."
    },
    "actions_took":
    [
        {
            "tool": "get_item_spec",
            "tool_input":
            {
                "item_name": "toy bear"
            },
            "tool_output":
            {
                "name": "toy bear",
                "color": "red",
                "age_group": "1-5 years old"
            }
        },
        {
            "tool": "search_image_path_for_item",
            "tool_input":
            {
                "item_name": "toy bear"
            },
            "tool_output": "[images/toy.png]"
        }
    ]
}

```

## Possible failures
Although generative agents have achieved impressive performance and we are observing the "spark 
of artificial general intelligence", it is not uncommon to observe failures from them. Here are 
some possible failure cases users might run into
1. Repeated clarifying question    
   Although agent has access the past conversation history and tools outputs, it might still ask 
   for the same clarifying question again or not directly responding to the user when it has 
   sufficient information to do so. For example, user has already provided the order id, agent 
   might ask for order id again in some rare cases, or responds "Let me look it up for you" 
   instead of taking the action and respond with the answer. Usually, agent would have better 
   performance if prompt is more explicit.    
2. Use the same tool again  
   It is not very uncommon to see agent tries to use the same tool again with same inputs. 
   AutoChain framework prevents user from doing exactly the same action again and forces agent 
   to respond to user in that case. 
3. Invalid JSON output  
   JSON can be a finicky format to deal with. `AgentOutputParser` will try to correct the JSON 
   output if it failed to load. However, JSON or similar structured format is still better to 
   parse and understand than free formed responses.