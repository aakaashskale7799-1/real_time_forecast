import schedule
import time
import subprocess
from loguru import logger

def retrain_job():
    logger.info("Running scheduled retraining job...")
    result = subprocess.run(["python", "run_train.py"], capture_output=True, text=True)
    if result.returncode == 0:
        logger.success("Retraining completed successfully.")
    else:
        logger.error(f"Retraining failed:\n{result.stderr}")

if __name__ == "__main__":
    logger.add("logs/retrain.log", rotation="10 MB", retention="7 days")

    # Schedule retraining (every day at 6 AM)
    schedule.every().day.at("06:00").do(retrain_job)

    logger.info("⏰ Retraining scheduler started. Waiting for trigger...")

    while True:
        schedule.run_pending()
        time.sleep(60)
