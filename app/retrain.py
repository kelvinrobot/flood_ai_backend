# app/retrain.py

import pandas as pd
import torch
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.model_fusion import LSTMNet, autoencoder_model

#  Retrain CatBoost 
def retrain_catboost(data: pd.DataFrame):
    model = CatBoostClassifier(verbose=0)
    X = data.drop(columns=["flood"])
    y = data["flood"]
    model.fit(X, y)
    model.save_model("models/catboost_flood_model.cbm")

#  Retrain XGBoost without using StandardScaler
def retrain_xgb(data: pd.DataFrame):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    X = data.drop(columns=["flood_risk_score"])
    y = data["flood_risk_score"]

    model.fit(X, y)
    joblib.dump(model, "models/xgboost_flood_risk_model.pkl")

    print("XGBoost retrained successfully (no scaler used)")

# Retrain LSTM 
def retrain_lstm(data: pd.DataFrame):
    input_cols = [
        "avg_temp", "humidity", "precip", "windspeed", "sealevelpressure",
        "cloudcover", "solarradiation", "flood_lag_1", "flood_lag_2",
        "flood_lag_3", "flood_lag_4", "flood_lag_5", "smi_linear_norm", "severerisk", 'month'
    ]
    X = data[input_cols].values
    y = data["flood"].values

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = LSTMNet(input_size=X.shape[1])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "models/lstm_flood_model_weights.pth")

# Retrain Autoencoder 
def retrain_autoencoder(data: pd.DataFrame):
    input_cols = [
        "avg_temp", "humidity", "precip", "windspeed", "sealevelpressure",
        "cloudcover", "solarradiation", "flood_lag_1", "flood_lag_2",
        "flood_lag_3", "flood_lag_4", "flood_lag_5", "smi_linear_norm"
    ]
    X = data[input_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 8)
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, X.shape[1])
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model, "models/flood_autoencoder_weights.pth")
    joblib.dump(scaler, "models/autoencoder_scaler.pkl")

# Trigger all retrains 
def retrain_all_models(data: pd.DataFrame):
    retrain_catboost(data)
    retrain_xgb(data)
    retrain_lstm(data)
    retrain_autoencoder(data)
    print("All models retrained and saved.")
