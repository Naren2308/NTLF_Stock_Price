# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

# Initialize the Flask app
app = Flask(__name__)

# Load the scaler and the model
scaler = joblib.load('scaler.pkl')
model = load_model('lstm_model.h5')

# Load the data for plotting
data = pd.read_csv('NFLX.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preprocessing and feature engineering (same as the training script)
data = data[['Close']]
data['5_MA'] = data['Close'].rolling(window=5).mean()
data['30_MA'] = data['Close'].rolling(window=30).mean()
data['Volatility'] = data['Close'].rolling(window=5).std()
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Scale the data (normalization)
data_scaled = scaler.transform(data)

# Prepare data for predictions (the test data)
lookback = 90
X_test, y_test_actual = [], []

for i in range(lookback, len(data_scaled)):
    X_test.append(data_scaled[i-lookback:i])
    y_test_actual.append(data_scaled[i, 0])

X_test, y_test_actual = np.array(X_test), np.array(y_test_actual)
y_test_actual = scaler.inverse_transform(np.concatenate((y_test_actual.reshape(-1, 1), np.zeros((y_test_actual.shape[0], 4))), axis=1))[:, 0]

# Get predictions
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(np.concatenate((test_predictions, np.zeros((test_predictions.shape[0], 4))), axis=1))[:, 0]

# Route for rendering the homepage
@app.route('/')
def index():
    return render_template('index.html')

# API route to get predictions
@app.route('/api/predictions')
def predictions():
    dates = data.index[lookback:].strftime('%Y-%m-%d').tolist()
    actual_prices = y_test_actual.tolist()
    predicted_prices = test_predictions.tolist()
    return jsonify({'dates': dates, 'actual': actual_prices, 'predicted': predicted_prices})

if __name__ == '__main__':
    app.run(debug=True)
