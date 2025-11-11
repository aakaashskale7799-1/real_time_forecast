from prophet import Prophet
import pandas as pd
import joblib
from loguru import logger
from .base_model import BaseModel
#done
class ProphetModel(BaseModel):
    """Facebook Prophet time series model."""

    def __init__(self):
        super().__init__(model_name="Prophet")
        self.model = Prophet()

    def train(self, df: pd.DataFrame):
        """Train the model on historical data."""
        df_train = df.rename(columns={"timestamp": "ds", "value": "y"})
        logger.info("Training Prophet model...")
        self.model.fit(df_train)
        logger.success("Prophet model training complete.")

    def forecast(self, periods: int, freq: str = "H") -> pd.DataFrame:
        """Forecast into the future."""
        logger.info(f"Forecasting next {periods} periods...")
        future = self.model.make_future_dataframe(periods=periods, freq=freq.lower())
        forecast = self.model.predict(future)
        forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        forecast.rename(columns={"ds": "timestamp", "yhat": "forecast"}, inplace=True)
        logger.success("Forecasting complete.")
        return forecast

    def save(self, path: str):
        joblib.dump(self.model, path)
        logger.info(f"Model saved at {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
