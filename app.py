from flask import Flask, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__)

# Load model and scaler
model = load_model("C:/Users/Administrator/Documents/NLFX-Stock-Prediction/netflix_lstm_model.h5")
data = pd.read_csv('C:/Users/Administrator/Documents/NLFX-Stock-Prediction/NFLX.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[['Close']])

def prepare_data():
    # Apply data preparation steps as in your code, returning data for frontend
    # ...
    return {
        "train_dates": list(train_dates),
        "test_dates": list(test_dates),
        "train_actual": list(y_train_actual),
        "train_predicted": list(train_predictions),
        "test_actual": list(y_test_actual),
        "test_predicted": list(test_predictions)
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics")
def metrics():
    return jsonify({
        "train_mse": train_mse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2
    })

@app.route("/data")
def data():
    return jsonify(prepare_data())

app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    app.run(debug=True)
