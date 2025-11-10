from src.configurations_mlflow import mlflow_exp,mlflow_client
from src.configurations_mlflow import load_tokenizer_name

# lets get started with setting experiment name and client within the mlflow workflow:
mlflow_exp()

client=mlflow_client()

# ge the model's parameters:
model_name=load_tokenizer_name()
model_version="1"

client.transition_model_version_stage(
    name=model_version,
    version=model_version,
    stage="production"
)