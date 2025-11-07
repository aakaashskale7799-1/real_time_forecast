from pathlib import Path
import os
from datetime import datetime
import pandas as pd

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def timestamped_path(dir_path: str, prefix: str = "data"):
    ensure_dir(dir_path)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path(dir_path) / f"{prefix}_{ts}.parquet"

def atomic_write_parquet(df: pd.DataFrame, path: str):
    """
    Write parquet atomically: write to temp file then replace target.
    """
    tmp = Path(path + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(Path(path))
