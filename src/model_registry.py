import os
import shutil
from datetime import datetime
from loguru import logger
from pathlib import Path

class ModelRegistry:
    """Handles model versioning and best-model management."""

    def __init__(self, model_dir="models/", registry_dir="models/registry/"):
        self.model_dir = Path(model_dir)
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(self, model_path: str, metrics: dict):
        """Version the model with timestamp and metrics."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.registry_dir / f"model_{version}.pkl"
        shutil.copy(model_path, dest)

        info_path = self.registry_dir / f"model_{version}_metrics.txt"
        with open(info_path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        logger.success(f"Model version saved: {dest}")
        return dest
