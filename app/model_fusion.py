import torch
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost as xgb

from app.schemas import FloodRequest, FloodResponse

# IMPORTANT: these must match what you used when training CatBoost
CATBOOST_COLUMNS = [
    "Average temp", "humidity", "precip", "windspeed", "sealevelpressure",
    "cloudcover", "solarradiation", "severerisk",
    "flood_lag_1", "flood_lag_2", "flood_lag_3", "flood_lag_4", "flood_lag_5",
    "SMI_linear_norm", "month"
]

# Define LSTM model
class LSTMNet(torch.nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2):
        super(LSTMNet, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.softmax(out)

# Define Autoencoder model
class Autoencoder(torch.nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Load all models and scalers
catboost_model = CatBoostClassifier()
catboost_model.load_model("models/catboost_flood_model .cbm")

xgb_model = joblib.load("models/xgboost_flood_risk_model.pkl")

lstm_model = LSTMNet(input_size=15)
lstm_model.load_state_dict(torch.load("models/lstm_flood_model_weights.pth", map_location=torch.device('cpu')))
lstm_model.eval()

autoencoder_model = Autoencoder(input_size=15)
autoencoder_model.load_state_dict(torch.load("models/flood_autoencoder_weights.pth", map_location=torch.device('cpu')))
autoencoder_model.eval()

autoencoder_scaler = joblib.load("models/autoencoder_scaler.pkl")


# Predict logic
def predict_flood(data: FloodRequest) -> FloodResponse:
    # Convert Pydantic input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    input_df.columns = CATBOOST_COLUMNS

    print("\n Incoming request data for prediction:")
    print(input_df)

    features = input_df.values.astype(np.float32)
    model_votes = []

    #  CatBoost Prediction with probability
    cat_probs = catboost_model.predict_proba(input_df)[0]
    cat_prob = float(cat_probs[1])
    cat_pred = 1 if cat_prob > 0.4 else 0
    model_votes.append(f"CatBoost prob: {cat_prob:.2f} yes/no {'yes' if cat_pred else 'no'}")

    #  XGBoost risk score
    xgb_score = float(xgb_model.predict(features)[0])
    model_votes.append(f"XGBoost risk_score: {xgb_score:.2f}")

    # LSTM prediction
    lstm_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)
    lstm_out = lstm_model(lstm_input)  # (1, 3)
    lstm_pred_class = int(torch.argmax(lstm_out).item())
    lstm_pred = 1 if lstm_pred_class > 0 else 0
    lstm_prob = float(lstm_out[0][lstm_pred_class].item())
    model_votes.append(f"LSTM: {'yes' if lstm_pred else 'no'} (class={lstm_pred_class}, conf={lstm_prob:.2f})")

    #  Autoencoder anomaly insight
    ae_input = autoencoder_scaler.transform(features)
    ae_input = torch.tensor(ae_input, dtype=torch.float32)
    ae_recon = autoencoder_model(ae_input)
    ae_loss = torch.mean((ae_input - ae_recon) ** 2).item()
    model_votes.append(f"Autoencoder MSE: {ae_loss:.5f}")

    #  Final Flood Decision
    final_flood = int((cat_pred + lstm_pred) >= 1)

    #  Severity Label from risk score
    if xgb_score < 0.3:
        severity = "low"
    elif xgb_score < 0.6:
        severity = "mid"
    else:
        severity = "high"

    return FloodResponse(
        flood_probability_percent=round(((cat_prob + lstm_prob) / 2) * 100, 2),
        flood_risk_score_percent=round(xgb_score * 100, 2),
        severity_class=severity,
        model_votes=model_votes,
        final_flood=bool(final_flood)
    )
