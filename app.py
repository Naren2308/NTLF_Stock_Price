from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import joblib

app = Flask(__name__)

# Load the dataset and model (Ensure the model is saved with joblib or any appropriate method)
data = pd.read_csv('NFLX.csv')

# Data Preprocessing and Feature Engineering
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]
data['5_MA'] = data['Close'].rolling(window=5).mean()
data['30_MA'] = data['Close'].rolling(window=30).mean()
data['Volatility'] = data['Close'].rolling(window=5).std()
data['Returns'] = data['Close'].pct_change()

data.dropna(inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare the data for the LSTM model
lookback = 90
X, y = [], []

for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i-lookback:i])
    y.append(data_scaled[i, 0])  # 'Close' price is the target variable

X = np.array(X)
y = np.array(y)

# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=60, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=32)

# Define a route for rendering the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to return prediction data as JSON
@app.route('/predict')
def predict():
    # Predict on the test data or future data
    predictions = model.predict(X)
    
    # Inverse transform predictions
    predictions_rescaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 4))), axis=1))[:, 0]
    
    # Prepare data for the front end (dates and predictions)
    dates = data.index[lookback:].strftime('%Y-%m-%d').tolist()
    return jsonify({
        'dates': dates,
        'predictions': predictions_rescaled.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
as json_file:
    json.dump(prediction_data, json_file)

print("Predictions saved to predictions.json")
