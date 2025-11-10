from kfp import dsl,compiler
from kfp.dsl import pipeline,component,Input,Output,Artifact

## This is a kubeflow pipeline workflow for text generation:

@component
def text_generation_component(text):
    
    """A function for using both the tokenizer and model to generate text"""
    
    ## import the libs inside the workflow first:
    ###this will load the pipeline that we created for accessing artifacts without the need for joblib
    from serving import serving_pipeline
    from src.chat_with_model import response_model

    artifacts=serving_pipeline()
    
    ## get the model and tokenizer now:
    model=artifacts.outputs['model_artifact']

    tokenizer=artifacts.output['tokenizer_artifact']

    ## generate some text now:
    generated_text=response_model(prompt=text,model=model,tokenizer=tokenizer)

    return generated_text

## create the pipeline
@pipeline(
    name="llm-text-generation-pipeline",
    description="this is the pipeline for text generation"
)
def text_generation_pipeline():

    generated_text=text_generation_component()

    return generated_text

## compile the pipeline:
compiler.Compiler().compile(
    pipeline_func=text_generation_pipeline,
    package_path="pipelines/kubeflow_pipelines/text_generation_pipeline.yaml"
)