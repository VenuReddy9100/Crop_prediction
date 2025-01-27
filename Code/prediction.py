# model_usage.py

'''
# This module is responsible for loading the trained machine learning models and using them for predictions.
# It includes functions to:
  - Load the crop classification model (Voting Classifier)
  - Load the yield prediction model (RandomForestRegressor)
  - Accept user input for predictions
  - Make predictions using the loaded models for crop recommendation and yield prediction

# Libraries:
  - joblib (for loading models)
  - pandas (for handling input data)
'''

import joblib
import pandas as pd

# Function to load the classification model (crop recommendation)
def load_classification_model(filename):
    """Loads the trained classification model (Voting Classifier)"""
    model = joblib.load(filename)
    print(f"Loaded classification model...")
    return model

# Function to load the regression model (yield prediction)
def load_regression_model(filename):
    """Loads the trained regression model (RandomForestRegressor)"""
    model = joblib.load(filename)
    print(f"Loaded regression model...")
    return model

# Function to predict crop based on input features
def predict_crop(model, input_data):
    """Predicts the crop type based on the input features."""
    prediction = model.predict(input_data)
    return prediction

# Function to predict yield based on input features
def predict_yield(model, input_data):
    """Predicts the yield (ton per hectare) based on the input features."""
    prediction = model.predict(input_data)
    return prediction

# Accept user input and make predictions
def user_input():
    """Accepts user input for crop classification and yield prediction."""
    print("Enter the features for crop recommendation and yield prediction:")

    # For crop recommendation (input features: N, P, K, temperature, humidity, ph, rainfall)
    N = float(input("Enter Nitrogen content (N): "))
    P = float(input("Enter Phosphorous content (P): "))
    K = float(input("Enter Potassium content (K): "))
    temperature = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    ph = float(input("Enter pH level: "))
    rainfall = float(input("Enter Rainfall: "))

    # Create a DataFrame from user input for prediction
    crop_input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Predict the crop recommendation
    crop_model = load_classification_model('Models/voting_classifier_crop_model.joblib')
    crop_prediction = predict_crop(crop_model, crop_input_data)
    print(f"\nRecommended Crop: {crop_prediction[0]}")

    # For yield prediction (input features: N, P, K, pH, rainfall, temperature)
    yield_model = load_regression_model('Models/random_forest_yield_model.joblib')
        # Predicting the yield for the input
    yeild_input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'pH': [ph],
        'rainfall': [rainfall],
        'temperature': [temperature]
    })
    yield_prediction = predict_yield(yield_model, yeild_input_data)
    print(f"Predicted Yield: {yield_prediction[0]: .2f} tons per hectare")

if __name__ == "__main__":
    user_input()
