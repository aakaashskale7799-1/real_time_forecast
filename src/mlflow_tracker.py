import mlflow
from loguru import logger
from datetime import datetime
import os

class MLflowTracker:
    """Handles experiment tracking using MLflow."""

    def __init__(self, experiment_name="TimeSeriesForecasting"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("file:./mlruns")  # local folder tracking
        mlflow.set_experiment(experiment_name)

    def log_metrics(self, metrics: dict, model_path: str):
        with mlflow.start_run(run_name=f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(model_path)
            logger.success("Metrics and model logged to MLflow.")
