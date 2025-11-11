import pandas as pd
import yfinance as yf
from loguru import logger
from typing import Optional
# 8077439702
class DataFetcher:
    def __init__(self, symbol: str, interval: str = "1h", start_date: Optional[str] = None):
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date

    def fetch(self) -> pd.DataFrame:
        """
        Fetch OHLC data using yfinance and return DataFrame with
        columns ['timestamp', 'value'] where 'value' is Close price.
        """
        logger.info(f"Fetching {self.symbol} interval={self.interval} start={self.start_date}")
        df = yf.download(self.symbol, start=self.start_date, interval=self.interval, progress=False)
        if df.empty:
            logger.warning("yfinance returned empty DataFrame.")
            return pd.DataFrame(columns=["timestamp", "value"])
        df = df.reset_index()[["Datetime", "Close"]]
        df.columns = ["timestamp", "value"]
        # normalize timestamps: to UTC and drop tzinfo (store naive UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        logger.info(f"Fetched {len(df)} rows: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        return df