import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

rf_classifier = joblib.load('models/random_forest_classifier_model.pkl')
rf_regressor = joblib.load('models/random_forest_regressor_model.pkl')
lstm_model = load_model('models/lstm_temp_max_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_weather')
def predict_weather():
    return render_template('predict_weather.html')

@app.route('/predict_temperature')
def predict_temperature():
    return render_template('predict_temperature.html')

@app.route('/predict_time_series')
def predict_time_series():
    return render_template('predict_time_series.html')

@app.route('/process_time_series', methods=['POST'])
def process_time_series():
    period = request.form['period']
    sequence_length = 7 if period == '1_week' else 30 if period == '1_month' else None
    if not sequence_length:
        return "Invalid time period selected", 400

    dummy_data = np.random.rand(1, sequence_length, 1)  # Simplified data creation
    predictions = lstm_model.predict(dummy_data).flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(predictions)
    plt.xlabel('Day')
    plt.ylabel('Predicted Temp Max')
    plt.title(f'Predicted Temp Max for {period}')
    plt.grid(True)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_url = base64.b64encode(img.getvalue()).decode()

    return render_template('results.html', time_series_result=f'Predicted Temp Max for {period}', img_url=img_url, predicted_values=predictions.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    precipitation = float(request.form['precipitation'])
    wind = float(request.form['wind'])
    day_of_week = int(request.form['day_of_week'])
    month = int(request.form['month'])
    year = int(request.form['year'])

    if 'temp_max' in request.form and 'temp_min' in request.form:
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        input_features = pd.DataFrame([[precipitation, temp_max, temp_min, wind, day_of_week, month, year]], columns=['precipitation', 'temp_max', 'temp_min', 'wind', 'day_of_week', 'month', 'year'])
        prediction = rf_classifier.predict(input_features)[0]
        class_labels = ['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow']
        prediction_class = class_labels[prediction]
        return render_template('results.html', classifier_result=f'Predicted Weather: {prediction_class}')
    else:
        input_features = pd.DataFrame([[precipitation, wind, day_of_week, month, year]], columns=['precipitation', 'wind', 'day_of_week', 'month', 'year'])
        prediction = rf_regressor.predict(input_features)[0]
        return render_template('results.html', regressor_result=f'Predicted Temp Max: {prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
