import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import json

# Load model and dataset
model = load_model("netflix_lstm_model.h5")
data = pd.read_csv("NFLX.csv")

# Data preprocessing (ensure it matches what you used for training)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]
data['5_MA'] = data['Close'].rolling(window=5).mean()
data['30_MA'] = data['Close'].rolling(window=30).mean()
data['Volatility'] = data['Close'].rolling(window=5).std()
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare data
lookback = 90
X = []

for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i - lookback:i])

X = np.array(X)

# Make predictions
predictions = model.predict(X)
predictions_actual = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 4))), axis=1))[:, 0]

# Get dates for predictions
dates = data.index[lookback:]

# Save to JSON file
output_data = {
    "dates": [str(date) for date in dates],
    "predictions": predictions_actual.tolist()
}

with open("output.json", "w") as f:
    json.dump(output_data, f)

print("Predictions saved to output.json")
