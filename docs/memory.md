# Memory

We have a simple memory interface to experiment with. Memory is accessible at the `Chain` level,
and only at th `Chain` level, since it is the only stateful component. By default, memory saves
conversation history, including the latest user query, and intermediate
steps, which are `AgentAction` taken with corresponding outputs.

`Chain` could collect all the memory and puts into `inputs` at `prep_inputs` step and updates
memory at `prep_outputs` step. Constructed `inputs` will be passed to agent as kwargs.

There are two parts of the memory, conversation history and key-value memory

## Conversation history

Memory uses `ChatMessageHistory` to store all the conversation history between agent and user
as instances of `BaseMessage`, including `FunctionMessage`, which is tool used and
corresponding output. This make tracking all interactions easy and fit the same
interface OpenAI API requires.

## Key-value memory

Not only we could save conversation history, it allows saving any memory in key value pair
format. By default, it saves all the `AgentActions` and corresponding outputs using key value
pairs. This part is designed to be flexible and users could save anything to it with preferred
storage. One way is using this as long term memory powered by internal search tool. Example
implementation of it is under `autochain/memory/long_term_memory.py`. In that example, if the
value is an instance of document, it would be saved to `long_term_memory` and can be retrieved
using key as query.

## Types of memory supported

AutoChain supports different types of memory for different use cases.

### BufferMemory

This is the simplest implementation of memory. Everything stored in RAM with python dictionary
as key-value store. This is best suited for experimentation and iterating prompts, which is the
default type of memory AutoChain uses in examples and evaluation.

### LongTermMemory

In the case there are a lot of information need to be stored and only a small part of it is
needed during the planning step, `LongTermMemory` enables agents to retrieve partial memory
with internal search tool, such as `ChromaDBSearch`, `PineconeSearch`, `LanceDBSearch`. Search query is the 
key of the store, and it still follow the same interface as other memory implementations. Both 
would encode the text into vector DB and retrieve using the search query.

### RedisMemory

Redis is also supported to save information. This is useful when hosting AutoChain as a backend
service on more than one server instance, in which case it's not possible to use RAM as memory.
