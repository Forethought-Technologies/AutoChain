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