import os
import numpy as np
import pandas as pd
import torch
from train import StockLSTM

def load_model(ticker="AAPL", seq_len=60):
    # ----------------------------
    # Load Dataset
    # ----------------------------
    if not os.path.exists("all_stocks_5yr.csv"):
        raise FileNotFoundError("❌ Dataset file 'all_stocks_5yr.csv' not found!")

    df = pd.read_csv("all_stocks_5yr.csv")
    stock = df[df['Name'] == ticker]['close'].values.reshape(-1, 1)

    if len(stock) < seq_len:
        raise ValueError(f"❌ Not enough data for '{ticker}' to make a prediction.")

    # ----------------------------
    # Load Scaler
    # ----------------------------
    scaler_min_file = f"{ticker}_scaler.npy"
    scaler_scale_file = f"{ticker}_scale.npy"

    if not (os.path.exists(scaler_min_file) and os.path.exists(scaler_scale_file)):
        raise FileNotFoundError(f"❌ Scaler files for {ticker} not found. Train the model first.")

    min_ = np.load(scaler_min_file)
    scale_ = np.load(scaler_scale_file)

    def scale(data):
        return (data - min_) / scale_

    def inverse_scale(data):
        return data * scale_ + min_

    # ----------------------------
    # Load Trained Model
    # ----------------------------
    model_path = f"./saved_models/{ticker}_lstm.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}. Train the model first.")

    model = StockLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # ----------------------------
    # Prepare Last Sequence
    # ----------------------------
    last_seq = stock[-seq_len:]
    last_seq = scale(last_seq)
    last_seq = torch.tensor(last_seq.reshape(1, seq_len, 1), dtype=torch.float32)

    # ----------------------------
    # Make Prediction
    # ----------------------------
    with torch.no_grad():
        pred = model(last_seq).numpy()

    # Get current price (last actual price)
    current_price = float(stock[-1][0])
    
    # Ensure prediction is positive and reasonable
    predicted_price = float(inverse_scale(pred)[0][0])
    # Ensure prediction is positive and within a reasonable range of current price
    predicted_price = max(predicted_price, current_price * 0.5)  # Not less than 50% of current price
    
    return predicted_price, current_price

if __name__ == "__main__":
    predicted_price, current_price = load_model("AAPL")
    print(f"Current price: ${current_price:.2f}")
    print(f"📈 Predicted next closing price: ${predicted_price:.2f}")
