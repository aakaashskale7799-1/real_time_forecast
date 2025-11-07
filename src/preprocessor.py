from loguru import logger
import pandas as pd
from typing import Optional
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, dropna: bool = True, resample_rule: Optional[str] = None):
        self.dropna = dropna
        self.resample_rule = resample_rule
        self.scaler = MinMaxScaler()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Preprocessor.clean got empty df.")
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        if self.dropna:
            before = len(df)
            df = df.dropna(subset=["value"])
            logger.info(f"Dropped {before - len(df)} NA rows from 'value'.")
        if self.resample_rule is not None:
            df = df.set_index("timestamp").resample(self.resample_rule).agg({"value": "last"})
            df["value"] = df["value"].ffill(limit=3)
            df = df.reset_index()
            logger.info(f"Resampled to {self.resample_rule}; rows={len(df)}")
        return df

    def get_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["scaled_value"] = self.scaler.fit_transform(df[["value"]])
        logger.info("Built 'scaled_value' column using MinMaxScaler.")
        return df
