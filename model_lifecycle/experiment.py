import mlflow
from src.configurations_mlflow import load_host,load_exp_name,load_tags,mlflow_client

# create the variables using the imported functions:
host=load_host()
exp_name=load_exp_name()
tags=load_tags()

# get the client:
client=mlflow_client()

# create the phi1 experiment:
phi1_exp=client.create_experiment(
    name=exp_name,
    tags=tags
)


