import mlflow
from src.configurations_mlflow import test_model_registry,load_tokenizer_name
from src.configurations_mlflow import mlflow_exp,mlflow_tracker

# set the tracking uri within the mlfloow:
mlflow_tracker()
mlflow_exp()

# model parameters:
tokenizer_name=load_tokenizer_name()
tokenizer_version="1"

tokenizer_test_reg=test_model_registry(name=tokenizer_name,version=tokenizer_version)

print(tokenizer_test_reg)