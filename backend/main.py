from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict import load_model  # Changed from train to predict
from datetime import datetime
import numpy as np
import torch

app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        # Get prediction and current price
        predicted_price, current_price = load_model(ticker)
        
        # Calculate trend and confidence
        trend = 'up' if predicted_price > current_price else 'down'
        price_diff_percent = abs((predicted_price - current_price) / current_price * 100)
        confidence = min(95, max(60, 75 + price_diff_percent))  # Scale confidence between 60-95
        
        predicted_date = (datetime.now()).strftime("%Y-%m-%d")

        # Must match frontend keys exactly (camelCase)
        return {
            "predictedPrice": predicted_price,
            "trend": trend,
            "confidence": int(confidence),
            "predictedDate": predicted_date
        }

    except Exception as e:
        return {
            "predictedPrice": 0,
            "trend": "down",
            "confidence": 0,
            "predictedDate": datetime.now().strftime("%Y-%m-%d"),
            "error": str(e)
        }
