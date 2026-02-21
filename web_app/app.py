import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- 1. Load Models and Scaler ---
try:
    scaler = joblib.load('../models/scaler.pkl')
    models = {
        'RandomForest': joblib.load('../models/randomforest_model.pkl'),
        'LogisticRegression': joblib.load('../models/logisticregression_model.pkl'),
        'KNeighbors': joblib.load('../models/kneighbors_model.pkl')
    }
except FileNotFoundError as e:
    print(f"ERROR loading model files: {e}. Ensure models are in the '../models/' directory.")
    exit()

MODEL_METRICS = {
    'RandomForest': {'Accuracy': 0.7786, 'Precision': 0.7308, 'Recall': 0.6049, 'F1-Score': 0.6611},
    'LogisticRegression': {'Accuracy': 0.7749, 'Precision': 0.7091, 'Recall': 0.5802, 'F1-Score': 0.6385},
    'KNeighbors': {'Accuracy': 0.7316, 'Precision': 0.6094, 'Recall': 0.4815, 'F1-Score': 0.5385}
}

FEATURE_ORDER = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/')
def home():
    """Renders the main prediction form page."""
    return render_template('index.html', model_metrics=MODEL_METRICS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the UI."""
    try:
        data = request.json
        model_name = data['model_name']
        input_data = [data[feature] for feature in FEATURE_ORDER]
        final_input = np.array([input_data])
        scaled_input = scaler.transform(final_input)
        model = models.get(model_name)
        if model is None:
            return jsonify({'error': 'Invalid model selected'}), 400
        probability = model.predict_proba(scaled_input)[0, 1]
        prediction_label = "Diabetic" if probability >= 0.5 else "Non-Diabetic"

        response = {
            'prediction_label': prediction_label,
            'diabetic_probability': f"{probability*100:.2f}%"
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    import webbrowser
    import threading
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)