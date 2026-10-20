# Tool

The ability to use tools makes the agent incredible more powerful as shown in LangChain and
AutoGPT. We follow the similar concept of tool in LangChain here as well.
All the tools in LangChain can be easily ported over to AutoChain since they follow very
similar interface.  
Tool is essentially an object that implements a `run` function that takes in a dictionary of
kwargs. Since input parsing can be reused, in most cases, user would just need to pass the
callable function to create a new tool, and LLM will generate the inputs on the fly when it
needs to use the tool. As the result, the interface for `Tool` is below:

- **func**  
Function callable will be called at the `run` function. It will automatically generate the
typing information when using `OpenAIFunctionsAgent`.

- **description**  
To make it easy and descriptive for LLM model to understand when it should use this tool, it
would be great to have a description for proper tool usage.

## Other optional parameters

- **name**  
Tool name as identifier for model specify which tool to use. If this is not provided, it will
be same as the `func` name. User might want to provide a more descriptive name for the tool if
the function name is not very obvious.  

- **arg_description**  
Function calling feature of OpenAI supports adding description for each argument. User could
pass a dictionary of arg name and description using `arg_description` parameter. They will be
formatted into the prompt when using `OpenAIFunctionsAgent`. 


## Tools included
### Parallel web search and page fetching

`ConversationalAgent.from_llm_and_tools(llm=llm)` and
`OpenAIFunctionsAgent.from_llm_and_tools(llm=llm)` now include `web_search` and
`web_fetch` when `tools` is omitted or `None`. These tools use
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
at `https://search.parallel.ai/mcp`. Search runs in Fast mode with no account or
API key required. Anonymous use has service limits; failures are returned to the
agent as tool output without switching providers.

The existing `ChatOpenAI` adapter uses OpenAI's pre-1.0 Python API. Use the
version recorded in `poetry.lock` (`openai==0.27.8`) for that adapter; the web
tools themselves do not require OpenAI or Parallel credentials.

Pass `tools=[]` to disable all tools and keep the agent offline apart from model
inference. Any explicit tool list remains authoritative, including Google search
or custom tools. Constructing an agent does not make network requests. Calling
these tools sends the query or URL to Parallel; only enable them when web access
is appropriate for your application.

To combine them with your own tools or adjust the per-request timeout:

```python
from autochain.tools.parallel_search import parallel_search_tools

tools = parallel_search_tools(timeout=20) + your_other_tools
agent = OpenAIFunctionsAgent.from_llm_and_tools(llm=llm, tools=tools)
```

The synchronous client uses Python's standard library, including on Python 3.8.
Each HTTP request has a 30-second timeout by default. A chain keeps one opaque
search conversation identifier in its memory across turns; clear the memory to
start a new conversation. Separate chains should use separate memory instances.
Standalone `Tool.run` calls each start a separate research run. Ctrl-C cancels
execution rather than converting the interruption to a tool error.
Custom memory implementations should keep `get_conversation_id()` separate from
semantic retrieval and call `reset_conversation_id()` in their `clear()` method.

### GoogleSearchTool
Migrated from LangChain, which is also an example for user to easily migrate any tool from 
LangChain if needed.  
User would need to provide `google_api_key` and `google_cse_id` to 
search google through API. This allows the agent to have access to search engine and other 
non-parametric information.  

### PineconeTool
Internal search tool that can be used for long term memory of the agent or looking up relevant 
information that does not exists from the Internet. Currently, AutoChain supports `Pinecone` as 
long term memory for the agent


### ChromaDBTool
Internal search tool that can be used for long term memory of the agent or looking up relevant
information that does not exists from the Internet. Currently, AutoChain supports `ChromaDB` as
long term memory for the agent.

### LanceDBTool
Internal search tool that can be used for long term memory of the agent or looking up relevant
information that does not exists from the Internet. Currently, AutoChain supports `ChromaDB` as
long term memory for the agent. LanceDBTool is serverless, and does not require any setup.
