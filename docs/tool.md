# Tool

The ability to use tools makes the agent incredible more powerful as shown in LangChain and
AutoGPT. We follow the similar concept of tool in LangChain here as well.
All the tools in LangChain can be easily ported over to AutoChain since they follow very 
similar interface.  
Tool is essentially an object that implements a `run` function that takes in a dictionary of
kwargs. Since input parsing can be reused, in most cases, user would just need to pass the
callable function to create a new tool, and LLM will generate the inputs on the fly when it
needs to use the tool. As the result, the interface for `Tool` is below:

**name**
Tool name as identifier for model specify which tool to use

**func**
Function callable will be called at the `run` function

**description**
To make it easy and descriptive for LLM model to understand when it should use this tool, it
would be great to have a description for proper tool usage.
