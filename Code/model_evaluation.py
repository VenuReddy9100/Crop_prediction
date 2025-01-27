# model_evaluation.py

'''
# This module evaluates the trained machine learning models for crop classification and yield prediction.
# It includes:
#   - Evaluating the performance of the Voting Classifier for crop recommendation
#   - Evaluating the RandomForestRegressor for yield prediction
#   - Printing classification report, confusion matrix, and accuracy for classification
#   - Printing mean squared error and r2 score for regression

# Libraries:
#   - scikit-learn (for evaluation metrics and model evaluation)
#   - seaborn & matplotlib (for visualization)
#   - pandas (for loading and manipulating data)
#   - joblib (for loading models)

'''

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Import functions from data_preprocessing.py
from data_preprocessing import load_crop_data, preprocess_crop_data, load_yield_data, preprocess_yield_data

def load_model(model_path):
    """Loads the trained model from the file."""
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_classification_model(model, X_test, y_test):
    """Evaluates the classification model."""
    # Predicting with the trained classification model
    y_pred = model.predict(X_test)
    
    # Accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Classification)')
    plt.show()

def evaluate_regression_model(model, X_test, y_test):
    """Evaluates the regression model."""
    # Predicting with the trained regression model
    y_pred = model.predict(X_test)
    
    # Mean Squared Error and R2 Score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Plotting actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs Predicted Yield')
    plt.show()

if __name__ == "__main__":
    # Load models and data
    classification_model = load_model('Models/voting_classifier_crop_model.joblib')
    regression_model = load_model('Models/random_forest_yield_model.joblib')
    
    # Loading and preprocessing crop data
    crop_data = load_crop_data('Data/Crop_recommendation.csv')  # --> Ensure correct path
    X_class, y_class = preprocess_crop_data(crop_data)  # Preprocessed data for classification
    
    # Split the data for classification
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.25, random_state=0)

    # Evaluate classification model
    if classification_model:
        print("\nEvaluating Classification Model:")
        evaluate_classification_model(classification_model, X_test_class, y_test_class)
    
    # Loading and preprocessing yield data
    yield_df = load_yield_data('Data/Dataset.xlsx')  # --> Ensure correct path
    X_yield, y_yield = preprocess_yield_data(yield_df)  # Preprocessed data for regression
    
    # Split the data for regression
    X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(X_yield, y_yield, test_size=0.25, random_state=0)
    
    # Evaluate regression model
    if regression_model:
        print("\nEvaluating Regression Model:")
        evaluate_regression_model(regression_model, X_test_yield, y_test_yield)
