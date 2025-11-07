import numpy as np
import pandas as pd
from loguru import logger

class Evaluator:
    """Evaluate forecasting model performance."""

    @staticmethod
    def rmse(actual: pd.Series, predicted: pd.Series) -> float:
        return np.sqrt(((actual - predicted) ** 2).mean())

    @staticmethod
    def mape(actual: pd.Series, predicted: pd.Series) -> float:
        mask = actual != 0
        return (np.abs((actual[mask] - predicted[mask]) / actual[mask]).mean()) * 100

    @classmethod
    def evaluate(cls, df: pd.DataFrame, actual_col="value", pred_col="forecast"):
        rmse = cls.rmse(df[actual_col], df[pred_col])
        mape = cls.mape(df[actual_col], df[pred_col])
        logger.info(f"RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        return {"RMSE": rmse, "MAPE": mape}
