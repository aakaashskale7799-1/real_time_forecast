import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from loguru import logger
from src.pipeline import ForecastPipeline

if __name__ == "__main__":
    logger.add("logs/pipeline.log", rotation="10 MB", retention="7 days")
    pipeline = ForecastPipeline("config.yaml")
    processed = pipeline.run()
    print(processed.tail(10))
