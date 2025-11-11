import pandas as pd
import os
from .mlflow_tracker import MLflowTracker
from loguru import logger
from .prophet_model import ProphetModel
from .evaluator import Evaluator
from .visualizer import Visualizer
from .model_registry import ModelRegistry
#working
class Trainer:
    """Handles full training workflow."""

    def __init__(self, model_dir="models/", forecast_horizon=24):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.forecast_horizon = forecast_horizon
        self.registry = ModelRegistry(model_dir=model_dir)

    def run(self, df: pd.DataFrame):
        logger.info("Initializing Prophet model...")
        model = ProphetModel()
        model.train(df)

        model_path = os.path.join(self.model_dir, "prophet_model.pkl")
        model.save(model_path)

        forecast = model.forecast(self.forecast_horizon)
        forecast = forecast.merge(df, on="timestamp", how="left")

        metrics = Evaluator.evaluate(forecast.dropna(subset=["value", "forecast"]))

        # Visualizations
        Visualizer.plot_static(forecast, save_path="models/forecast_plot.png")
        Visualizer.plot_interactive(forecast, save_path="models/forecast_plot.html")

        # Version the model
        self.registry.register(model_path, metrics)

        logger.success(f"Training complete. Metrics: {metrics}")

        tracker = MLflowTracker(experiment_name="ProphetForecasting")
        tracker.log_metrics(metrics, model_path)
        return forecast, metrics
