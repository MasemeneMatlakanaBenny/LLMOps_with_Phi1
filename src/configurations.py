import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_name=""
## create a function that will be used to set the device to cuda if available else cpu:
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    else:
        return torch.device("cpu")

## create a function that will be used to load the tokenizer from transformers:
def load_tokenizer(model_name:str=model_name)->AutoTokenizer:
    # load the tokenizer first from transformers module:
    tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

    return tokenizer

## create a function that will be used to load the model from transformers:
def load_model(model_name:str=model_name,device:torch.device=set_device())->AutoModelForCausalLM:
    # load the model from the transformers module:
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto"
    )

    return model.to(device)

