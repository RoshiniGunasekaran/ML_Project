from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
rf_classifier = joblib.load('models/random_forest_classifier_model.pkl')
rf_regressor = joblib.load('models/random_forest_regressor_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_weather')
def predict_weather():
    return render_template('predict_weather.html')

@app.route('/predict_temperature')
def predict_temperature():
    return render_template('predict_temperature.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    precipitation = float(request.form['precipitation'])
    wind = float(request.form['wind'])
    day_of_week = int(request.form['day_of_week'])
    month = int(request.form['month'])
    year = int(request.form['year'])

    # Check the form to decide which prediction to make
    if 'temp_max' in request.form and 'temp_min' in request.form:
        # Weather prediction
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        input_features = pd.DataFrame([[precipitation, temp_max, temp_min, wind, day_of_week, month, year]], 
                                      columns=['precipitation', 'temp_max', 'temp_min', 'wind', 'day_of_week', 'month', 'year'])
        prediction = rf_classifier.predict(input_features)[0]
        class_labels = ['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow']
        prediction_class = class_labels[prediction]
        return render_template('results.html', classifier_result=f'Predicted Weather: {prediction_class}')
    else:
        # Temperature prediction
        input_features = pd.DataFrame([[precipitation, wind, day_of_week, month, year]], 
                                      columns=['precipitation', 'wind', 'day_of_week', 'month', 'year'])
        prediction = rf_regressor.predict(input_features)[0]
        return render_template('results.html', regressor_result=f'Predicted Temp Max: {prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
