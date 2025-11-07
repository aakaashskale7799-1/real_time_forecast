from pathlib import Path
import pandas as pd
from loguru import logger
from src.utils import atomic_write_parquet, timestamped_path, ensure_dir

class DataStore:
    def __init__(self, raw_path: str = "data/raw/live_data.parquet", versioned_dir: str = "data/versioned/"):
        self.raw_path = Path(raw_path)
        self.versioned_dir = Path(versioned_dir)
        ensure_dir(self.raw_path.parent.as_posix())
        ensure_dir(self.versioned_dir.as_posix())

    def save(self, df: pd.DataFrame):
        """
        Overwrite canonical raw file atomically and write a versioned snapshot.
        """
        if df.empty:
            logger.warning("save() called with empty DataFrame; skipping save.")
            return
        atomic_write_parquet(df, self.raw_path.as_posix())
        snap = timestamped_path(self.versioned_dir.as_posix(), prefix="raw")
        df.to_parquet(snap, index=False)
        logger.info(f"Saved raw -> {self.raw_path.as_posix()}, snapshot -> {snap}")

    def append_incremental(self, df: pd.DataFrame):
        """
        Read existing raw (if present), merge with incoming df, deduplicate by timestamp,
        then save merged canonical file + snapshot.
        """
        if df.empty:
            logger.warning("append_incremental called with empty df; skipping.")
            return
        if not self.raw_path.exists():
            self.save(df)
            return
        existing = pd.read_parquet(self.raw_path)
        merged = pd.concat([existing, df]).drop_duplicates(subset=["timestamp"], keep="last")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        self.save(merged)
