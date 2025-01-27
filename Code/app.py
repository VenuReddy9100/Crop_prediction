from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
voting_clf = joblib.load('C:/Users/reddy/Desktop/Crop_Prediction/Crop_Prediction/Models/voting_classifier_crop_model.joblib')
rf_regressor = joblib.load('C:/Users/reddy/Desktop/Crop_Prediction/Crop_Prediction/Models/random_forest_yield_model.joblib')

# Home route to display the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to display the prediction form (GET) and handle form submission (POST)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')  # Show the form to the user

    # Get the data from the form when submitted
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Predict crop
    crop_prediction = voting_clf.predict(input_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])

    # Predict yield
    input_yield_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'pH': [ph],
        'rainfall': [rainfall],
        'temperature': [temperature]
    })
    yield_prediction = rf_regressor.predict(input_yield_data)
    formatted_yield_prediction = f"{yield_prediction[0]:.2f}"
    
    # Render the results on the result page
    return render_template('result.html', crop=crop_prediction[0], yield_pred=formatted_yield_prediction)

if __name__ == '__main__':
    app.run(debug=True)
