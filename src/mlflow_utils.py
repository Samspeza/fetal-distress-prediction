import os
import mlflow
from dotenv import load_dotenv

def config_mlflow():
    """
    Configura variáveis de ambiente e URI do MLflow com autenticação.
    """
    load_dotenv()  # Carrega variáveis do .env

    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
    
    mlflow.set_tracking_uri("https://dagshub.com/renansantosmendes/mlops-ead-2025.mlflow")
    
    mlflow.keras.autolog(log_models=True,
                         log_input_examples=True,
                         log_model_signatures=True)


def register_model(run_id: str, model_name: str = 'fetal_health'):
    """
    Registra o modelo treinado no MLflow.
    """
    run_uri = f"runs:/{run_id}/model"
    mlflow.register_model(run_uri, model_name)
