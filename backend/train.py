import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

# -------------------------
# LSTM Model
# -------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return F.softplus(out)  # ensures output > 0 but keeps small values

# -------------------------
# Load Model Function
# -------------------------
def load_model(ticker="AAPL"):
    model = StockLSTM()
    model_path = f"./saved_models/{ticker}_lstm.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load scaler parameters
    scaler_min = np.load(f"{ticker}_scaler.npy")
    scaler_scale = np.load(f"{ticker}_scale.npy")
    
    scaler = MinMaxScaler()
    scaler.min_ = scaler_min
    scaler.scale_ = scaler_scale
    
    return model, scaler

# -------------------------
# Training Function
# -------------------------
def train_model(ticker="AAPL", epochs=100, seq_len=60):
    os.makedirs("saved_models", exist_ok=True)

    # Load dataset
    df = pd.read_csv("all_stocks_5yr.csv")
    stock = df[df['Name'] == ticker]['close'].values.reshape(-1, 1)

    if len(stock) < seq_len:
        raise ValueError(f"❌ Not enough data for ticker '{ticker}' to train the model.")

    # Normalize data
    scaler = MinMaxScaler()
    stock_scaled = scaler.fit_transform(stock)

    # Prepare sequences
    X, y = [], []
    for i in range(len(stock_scaled) - seq_len):
        X.append(stock_scaled[i:i+seq_len])
        y.append(stock_scaled[i+seq_len])

    X, y = np.array(X), np.array(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Initialize model, loss, optimizer
    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # Save model and scaler
    model_path = f"./saved_models/{ticker}_lstm.pt"
    torch.save(model.state_dict(), model_path)
    np.save(f"{ticker}_scaler.npy", scaler.min_)
    np.save(f"{ticker}_scale.npy", scaler.scale_)

    print(f"✅ Model trained & saved at: {model_path}")
    print(f"✅ Scaler saved as: {ticker}_scaler.npy & {ticker}_scale.npy")

    return model, scaler

if __name__ == "__main__":
    train_model("AAPL")
