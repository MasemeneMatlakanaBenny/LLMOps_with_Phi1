from configurations import set_device,load_model,load_tokenizer
from chat_with_model import response_model
from flask import Flask
from schemas import Prompt
device=set_device() # set the device to cuda if available else cpu



# create the app
app=Flask(__name__)

# get the prompt from the user:
prompt=input("Enter the prompt here: ")

# validate the prompt we entered:
prompt_validated=Prompt(**prompt)



@app.route('/response from model')
def chatting_with_the_model(prompt):

    generated_text=response_model(prompt)

    return generated_text


@app.route('/metrics')
def evaluate_response(prompt):
    """This is a function that will be used to validate the prompt""" 
    
    import evaluate
    perplexity=evaluate.load("perplexity")

    generated_text=response_model(prompt)

    metrics=perplexity.compute(
        add_start_token=True,
        model_id="phi1",
        predictions=[generated_text],
        references=[prompt]
    )

    return metrics

