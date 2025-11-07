from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from loguru import logger
from src.prophet_model import ProphetModel

app = FastAPI(title="Time Series Forecast API")

class ForecastRequest(BaseModel):
    periods: int = 24
    freq: str = "h"

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("models/prophet_model.pkl")
        logger.success("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        model = None

@app.post("/forecast")
def forecast(req: ForecastRequest):
    if model is None:
        return {"error": "Model not available."}

    logger.info(f"Generating forecast for {req.periods} periods...")
    prophet_model = ProphetModel()
    prophet_model.model = model
    future = prophet_model.model.make_future_dataframe(periods=req.periods, freq=req.freq)
    forecast = prophet_model.model.predict(future)
    result = forecast[["ds", "yhat"]].tail(req.periods).rename(columns={"ds": "timestamp", "yhat": "forecast"})

    return {"forecast": result.to_dict(orient="records")}
