# Memory

We have a simple memory interface to experiment with. Memory is accessible at the `Chain` level,
and only at th `Chain` level, since it is the only stateful component. By default, memory saves
chat history, including the latest user query, and intermediate
steps, which are `AgentAction` taken with corresponding outputs.

`Chain` could collect all the memory and puts into `inputs` at `prep_inputs` step and updates
memory at `prep_outputs` step. Constructed `inputs` will be passed to agent as kwargs.

There are two parts of the memory, chat history and key-value memory

## Chat history

Memory uses `ChatMessageHistory` to store all the chat history between agent and user
as instances of `BaseMessage`, including `FunctionMessage`, which is tool used and 
corresponding output. This make tracking all interactions easy and fit the same 
interface OpenAI API requires.

## Key-value memory

Not only we could save chat history, it allows saving any memory in key value pair
format. By default, it saves all the `AgentActions` and corresponding outputs using key value
pairs. This part is designed to be flexible and users could save anything to it with preferred
storage. One way is using this as long term memory powered by internal search tool. Example
implementation of it is under `autochain/memory/long_term_memory.py`. In that example, if the
value is an instance of document, it would be saved to `long_term_memory` and can be retrieved
using key as query.