from src.auto_retrainer import AutoRetrainer

if __name__ == "__main__":
    retrainer = AutoRetrainer(config_path="config.yaml")
    retrainer.run_daily()
