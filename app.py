from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load('telecom_model.pkl')
scaler = joblib.load('telecom_scaler.pkl')  # Save scaler too!

@app.route('/')
def home():
    return render_template('abc.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert to df (match your features)
    df = pd.DataFrame([data])
    num_cols = ['tenure', 'MonthlyCharges']  # Scale only numerics
    df[num_cols] = scaler.transform(df[num_cols])
    # Contract is already encoded as int
    pred_proba = model.predict_proba(df)[0][1]
    risk = 'High' if pred_proba > 0.7 else 'Medium' if pred_proba > 0.4 else 'Low'
    return jsonify({'churn_prob': pred_proba, 'risk': risk})

if __name__ == '__main__':
    app.run(debug=True)