import mlflow
import os
from dotenv import load_dotenv

def config_mlflow():
    load_dotenv()
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
    mlflow.set_tracking_uri("https://dagshub.com/renansantosmendes/mlops-ead-2025.mlflow")
    mlflow.keras.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

def register_model(run_id, model_name='fetal_health'):
    run_uri = f"runs:/{run_id}/model"
    mlflow.register_model(run_uri, model_name)
