import matplotlib.pyplot as plt
from loguru import logger
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

class Visualizer:
    """Handles forecast visualization (matplotlib + plotly)."""

    @staticmethod
    def plot_static(df: pd.DataFrame, save_path: str = "models/forecast_plot.png"):
        """Simple matplotlib visualization of actual vs forecast."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df["timestamp"], df["value"], label="Actual", linewidth=2)
            plt.plot(df["timestamp"], df["forecast"], label="Forecast", linestyle="--", linewidth=2)
            plt.legend()
            plt.title("Actual vs Forecast")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True, linestyle="--", alpha=0.6)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Static plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")

    @staticmethod
    def plot_interactive(df: pd.DataFrame, save_path: str = "models/forecast_plot.html"):
        """Interactive Plotly visualization."""
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["forecast"], mode="lines", name="Forecast"))
            fig.update_layout(
                title="Interactive Forecast Visualization",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white"
            )
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Interactive plot failed: {e}")
