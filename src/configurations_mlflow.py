import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException

host="http://127.0.0.1:5000"

exp_name="LLMOPs with a family of Phi-1 Models"
exp_description="Using Phi1 which is a family of small language models"
tags={
    "project_name":"Phi1 Operations Modeling",
    "team":"AI/ML Team",
    "team lead":"Masemene Matlakana Benny",
    "date":"09 November 2025",
    "mlflow.note.content":exp_description,
    
}

# create the functions that we can reuse over and over again:
def load_host():
    return host

##create a function that will be reused to load the experiment name:
def load_exp_name():
    return exp_name

## create a function that will be used to get the experiment tags:
def load_tags():
    return tags

## create the client function for model versioning and staging
def mlflow_client(server=host)->MlflowClient:
    client=MlflowClient(tracking_uri=server)

    # return the client:
    return client


## create the function for mlflow.set_tracking_uri:
def mlflow_tracker(server=host):
    return mlflow.set_tracking_uri(server)


## create the function for mlflow.set_experiment_name:
def mlflow_exp(experiment_name=exp_name):
    return mlflow.set_experiment(name=experiment_name)


## function for the registered model name within mlflow
def load_model_name():
    return "phi1_model"

## function for the registered tokenizer model name within mlflow:
def load_tokenizer_name():
    return "phi1_tokenizer"

## function for testing model registry
def test_model_registry(name, version):
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, version=version)
        print(f"Model '{name}' version {version} exists")
    except RestException:
        print(f" Model '{name}' version {version} not found")
        

## a function for testing model versioning or stages per model
def test_model_versioning(name, stage):
    """
    Check if a specific model version exists in MLflow Model Registry
    and print the result.

    Args:
        name (str): Model name in the registry.
        version (str or int): Version number of the model.
    """
    client = mlflow_client()
    try:
        client.get_model_version(name=name, stage=stage)
        print(f"Model '{name}' at {stage} exists")
    except RestException:
        print(f"Model '{name}' at {stage} not found")