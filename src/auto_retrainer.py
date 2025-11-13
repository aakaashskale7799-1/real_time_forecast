import os
from loguru import logger
from datetime import datetime
from src.pipeline import ForecastPipeline
from src.trainer import Trainer

class AutoRetrainer:
    """
    Automates the entire ML workflow:
    - Fetch fresh data
    - Clean + preprocess
    - Train + evaluate
    - Save model + logs
    """

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        date = datetime.now().strftime("%Y-%m-%d")
        self.log_file = f"logs/run_{date}.log"

        os.makedirs("logs", exist_ok=True)
        logger.add(self.log_file)

    def run_daily(self):
        logger.info("Starting automated daily retraining job...")

        # 1. Load pipeline
        pipeline = ForecastPipeline(self.config_path)
        df = pipeline.run()

        # 2. Train model
        trainer = Trainer()
        forecast, metrics = trainer.run(df)

        logger.success(f"Daily automation complete. Metrics: {metrics}")
