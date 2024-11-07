import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json

# Load the dataset
data = pd.read_csv('NFLX.csv')

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Selection
data = data[['Close']]

# Feature Engineering
data['5_MA'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
data['30_MA'] = data['Close'].rolling(window=30).mean()  # 30-day moving average
data['Volatility'] = data['Close'].rolling(window=5).std()  # 5-day volatility
data['Returns'] = data['Close'].pct_change()  # Daily returns

# Drop NA values created by rolling calculations
data.dropna(inplace=True)

# Scale the data (normalization)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Train-test Split
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Prepare the data
lookback = 90
X_train, y_train = [], []
X_test, y_test = [], []

# Create the training dataset
for i in range(lookback, len(train_data)):
    X_train.append(train_data[i-lookback:i])
    y_train.append(train_data[i, 0])  # 'Close' is the first column

# Create the testing dataset
for i in range(lookback, len(test_data)):
    X_test.append(test_data[i-lookback:i])
    y_test.append(test_data[i, 0])  # 'Close' is the first column

# Convert lists to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# LSTM Model
model = Sequential()
model.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model and fit it
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform to get actual prices
train_predictions = scaler.inverse_transform(np.concatenate((train_predictions, np.zeros((train_predictions.shape[0], 4))), axis=1))[:, 0]
y_train_actual = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], 4))), axis=1))[:, 0]
test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, np.zeros((test_predictions.shape[0], 4))), axis=1))[:, 0]
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]

# Prepare the data for JSON
dates = data.index[lookback:].strftime('%Y-%m-%d').tolist()  # Dates of predictions
prediction_data = {
    "dates": dates,
    "train_predictions": train_predictions.tolist(),
    "test_predictions": test_predictions.tolist(),
    "y_train_actual": y_train_actual.tolist(),
    "y_test_actual": y_test_actual.tolist()
}

# Save to JSON file
with open('predictions.json', 'w') as f:
    json.dump(prediction_data, f)
    # Removing this line as json_file is not defined here and the file has already been closed. 
    #json.dump(prediction_data, json_file)

print("Predictions saved to predictions.json")
