from kfp import dsl,compiler
from kfp.dsl import pipeline,component,Input,Output,Artifact

@component
def serving_component(model_artifact:Output[Artifact],
                      tokenizer_artifact:Output[Artifact]):
    """This is a component for both model and tokenizer serving"""
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

    ## save the artifacts:
    joblib.dump(model,model_artifact.path)
    joblib.dump(tokenizer,model_artifact.path)

# create the pipeline for serving
@pipeline(
    name="llm-serving-pipeline",
    description="pipeline for model and tokenizer serving"
)
# create the function for the pipeline:
def serving_pipeline():
    """
    This is a pipeline for serving both the model and tokenizer after they have been moved to the 
    production stage in their lifecycle.
    We created the serving pipeline for loading and saving those model and tokenizer as artifacts
    """

    # served artifacts :
    served_artifacts=serving_component()

    return served_artifacts

# compile the pipeline and save it as a  yaml file:
compiler.Compiler().compile(
    pipeline_func=serving_pipeline,
    package_path="pipelines/kubeflow_pipelines/serving_pipeline.yaml"
)


    
