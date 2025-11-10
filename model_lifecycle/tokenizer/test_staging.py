import mlflow
from src.configurations_mlflow import test_model_versioning,load_model_name
from src.configurations_mlflow import mlflow_exp,mlflow_tracker

# set the tracking uri within the mlfloow:
mlflow_tracker()
mlflow_exp()

# model parameters:
tokenizer_name=load_model_name()
tokenizer_stage="staging"

tokenizer_test_staging=test_model_versioning(name=tokenizer_name,stage=tokenizer_stage)

print(tokenizer_test_staging)