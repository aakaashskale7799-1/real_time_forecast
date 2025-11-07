from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class BaseModel(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, df: pd.DataFrame):
        """Train the forecasting model."""
        pass

    @abstractmethod
    def forecast(self, periods: int) -> pd.DataFrame:
        """Generate future forecasts."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save trained model."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load trained model."""
        pass
