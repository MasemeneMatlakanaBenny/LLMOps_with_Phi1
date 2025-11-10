from prefect import task,flow


# model and tokenizer serving first:
@task
def serving_component():
      ## import libraries we have built in the workflow:
    import joblib
    import mlflow.pyfunc
    from src.configurations_mlflow import load_model_name,load_tokenizer_name
    from src.configurations_mlflow import mlflow_exp

    ## define the uri's parameters of both model and tokenizer-> name and stage
    model_name=load_model_name()
    tokenizer_name=load_tokenizer_name()
    stage="production"

    ## use the parameters to define the uri
    model_uri=f"models:/{model_name}/{stage}"
    tokenizer_uri=f"models:/{model_name}/{stage}"
    
    ## get both the model and tokenizer
    model=mlflow.pyfunc.load_model(model_uri=model_uri)
    tokenizer=mlflow.pyfunc.load_model(model_uri=tokenizer_uri)

    return model,tokenizer


@task
def text_generation_component(model,tokenizer,text):
     ## import the libs inside the workflow first:
    ###this will load the pipeline that we created for accessing artifacts without the need for joblib
    from src.chat_with_model import response_model

    ## generate some text now:
    generated_text=response_model(prompt=text,model=model,tokenizer=tokenizer)

    return generated_text

@task 
def model_evaluation(response:str):
       ## import libs
    import evaluate

    # use perplexity to evaluate the model:
    perplexity=evaluate.load("perplexity")

    # get the metrics:
    metrics=perplexity.compute(
        add_start_token=True,
        model_id="phi1",
        predictions=[response]
    )

    return metrics


@flow
def pipeline_workflow():
    model,tokenizer=serving_component()
    response=text_generation_component()
    metrics=model_evaluation()

    return metrics





