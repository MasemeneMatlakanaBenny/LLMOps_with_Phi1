
from kfp import dsl,compiler
from kfp.dsl import pipeline,component,Input,Output,Artifact


@component
def evaluation_component(response:str):
    
    """We evalaute the generate text"""

    ## import libs
    import evaluate
    from text_generation import text_generation_component

    ## define the generated text
    generated_text=text_generation_component()

    # use perplexity to evaluate the model:
    perplexity=evaluate.load("perplexity")

    # get the metrics:
    metrics=perplexity.compute(
        add_start_token=True,
        model_id="phi1",
        predictions=[response]
    )

    return metrics

# create the pipeline for evaluation:
@pipeline(
    name="model-evaluation-pipeline",
    description="pipeline for evaluating the generated text with the use of perplexity"
)
def evaluation_pipeline():
    
    metrics=evaluation_component()

    return metrics


## compile the pipeline and package it into a yaml file:
compiler.Compiler().compile(
    pipeline_func=evaluation_pipeline,
    package_path="pipelines/kubeflow_pipelines/evaluation_pipeline.yaml"

)
