## registering the model on mlflow:
import mlflow
from src.configurations import load_model
from src.configurations_mlflow import load_model_name,mlflow_tracker,mlflow_exp

#get the model and parameters-> name and version:
model=load_model()
model_name=load_model_name()

# set the mlflow experiment and tracker within the mlflow workflow:
mlflow_tracker()
mlflow_exp()


#register the model:
## create the run first:
run_name="phi_model_run"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.transformers.log_model(transformers_model=model,
                                  registered_model_name=model_name)

