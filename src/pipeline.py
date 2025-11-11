import yaml
from loguru import logger
from src.data_fetcher import DataFetcher
from src.data_store import DataStore
from src.preprocessor import Preprocessor
import pandas as pd
# done
class ForecastPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        data_cfg = cfg["data"]
        paths = cfg["paths"]
        pipeline_cfg = cfg.get("pipeline", {})

        self.fetcher = DataFetcher(symbol=data_cfg["symbol"], interval=data_cfg.get("interval", "1h"), start_date=data_cfg.get("start_date"))
        self.store = DataStore(raw_path=paths["raw"], versioned_dir=paths["versioned_dir"])
        self.preprocessor = Preprocessor(dropna=pipeline_cfg.get("dropna", True), resample_rule=pipeline_cfg.get("resample", None))

    def run(self) -> pd.DataFrame:
        logger.info("Pipeline run started.")
        df = self.fetcher.fetch()
        # Append safely to canonical raw store
        self.store.append_incremental(df)
        # Load canonical raw for processing
        raw = pd.read_parquet(self.store.raw_path)
        processed = self.preprocessor.clean(raw)
        logger.info("Pipeline run finished.")
        return processed
