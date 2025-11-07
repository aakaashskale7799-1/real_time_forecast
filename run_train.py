import os
import pandas as pd
from loguru import logger
from src.trainer import Trainer

if __name__ == "__main__":
    logger.add("logs/train.log", rotation="10 MB", retention="7 days")

    data_path = "data/raw/live_data.parquet"
    if not os.path.exists(data_path):
        logger.error(f"No data found at {data_path}. Run pipeline first.")
        exit()

    df = pd.read_parquet(data_path)
    df = df[["timestamp", "value"]]

    logger.info("Starting model training...")
    trainer = Trainer(model_dir="models/", forecast_horizon=24)
    forecast, metrics = trainer.run(df)

    logger.success("✅ Model training, evaluation, visualization, and versioning complete.")
    print(metrics)
