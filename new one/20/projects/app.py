from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
with open('D:\\PROJECT\\ML Project\\new one\\20\\temperature_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('D:\\PROJECT\\ML Project\\new one\\20\\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    current_year = datetime.now().year  # Get the current year

    if request.method == 'POST':
        user_month = request.form.get('month')
        user_year = request.form.get('year')

        # Validate user input
        if not user_month or not user_year:
            return render_template('index.html', error='Please enter both month and year.', current_year=current_year)

        try:
            user_year = int(user_year)
        except ValueError:
            return render_template('index.html', error='Invalid year input. Please enter a valid year.', current_year=current_year)

        if user_year < 1900 or user_year > current_year:
            return render_template('index.html', error='Invalid year input. Please enter a year between 1900 and the current year.', current_year=current_year)

        # Convert the user input month to a numerical value
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        user_month_num = month_map.get(user_month)

        if user_month_num is None:
            return render_template('index.html', error='Invalid month input. Please enter a valid month.', current_year=current_year)

        # Assuming predictions are for the first day of the given month/year
        day_of_year = datetime(user_year, user_month_num, 1).timetuple().tm_yday
        
        # Create a dataframe for the user input
        user_data = pd.DataFrame({
            'Month_Sin': [np.sin(2 * np.pi * user_month_num / 12.0)],
            'Month_Cos': [np.cos(2 * np.pi * user_month_num / 12.0)],
            'Day_Sin': [np.sin(2 * np.pi * day_of_year / 365.0)],
            'Day_Cos': [np.cos(2 * np.pi * day_of_year / 365.0)]
        })

        # Make a prediction using the model
        forecast = model.predict(user_data)

        # Inverse transform the forecast
        forecast_original = scaler.inverse_transform(forecast.reshape(-1, 1))

        return render_template('index.html', forecast=f'Forecasted Temperature: {forecast_original[0][0]:.2f}', current_year=current_year)
    
    return render_template('index.html', current_year=current_year)

if __name__ == '__main__':
    app.run(debug=True)