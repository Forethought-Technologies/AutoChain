# Models

## Huggingface text generation model
To use open sourced model from huggingface, AutoChain introduces 
`HuggingFaceTextGenerationModel` to support this use case.  
Requirements to be installed
```shell
transformers
torch
accelerate
```

Example usage
```python
from autochain.models.huggingface_text_generation_model import (
    HuggingFaceTextGenerationModel,
)
from autochain.agent.conversational_agent.conversational_agent import (
    ConversationalAgent,
)

llm = HuggingFaceTextGenerationModel(model_name="mosaicml/mpt-7b", 
                                     model_kwargs={"trust_remote_code":True})
agent = ConversationalAgent.from_llm_and_tools(llm=llm)
```
> Task planning could be a too challenging task for "small" model 