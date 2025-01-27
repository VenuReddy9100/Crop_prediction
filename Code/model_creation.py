# model_creation.py

''' 
# This module is responsible for training and saving the machine learning models used for crop classification and yield prediction.
# It includes:
  - Training a Voting Classifier for crop recommendation (RandomForest, KNN, GradientBoosting)
  - Training a RandomForestRegressor for yield prediction
  - Saving the trained models as joblib files for future use

# Libraries:
  - scikit-learn (for machine learning models)
  - joblib (for saving models)
  - os (for file saving)
'''

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import load_crop_data, preprocess_crop_data, load_yield_data, preprocess_yield_data
import os

def create_classification_model(X_train, y_train):
    """Creates and trains the classification model."""
    # Define the base models for classification
    rf_clf = RandomForestClassifier(random_state=1)
    knn_clf = KNeighborsClassifier()
    gb_clf = GradientBoostingClassifier()

    # Create a Voting Classifier (Hybrid Model) for classification
    voting_clf = VotingClassifier(
        estimators=[('Random Forest', rf_clf), ('KNN', knn_clf), ('Gradient Boosting', gb_clf)],
        voting='soft'
    )

    # Train the hybrid model for classification
    voting_clf.fit(X_train, y_train)
    print("Classification model (Voting Classifier) trained successfully.")
    return voting_clf

def create_regression_model(X_yield_train, y_yield_train):
    """Creates and trains the regression model for yield prediction."""
    rf_regressor = RandomForestRegressor(random_state=1)
    rf_regressor.fit(X_yield_train, y_yield_train)
    print("Regression model (RandomForestRegressor) trained successfully.")
    return rf_regressor

def save_model(model, filename):
    """Saves the trained model to a file."""
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created")

    joblib.dump(model, filename)
    print(f"Model saved as {filename}")



# Load and preprocess the crop data
crop_data = load_crop_data("Data/Crop_recommendation.csv")  # Path to crop recommendation data
# Features and target for classification
X, y = preprocess_crop_data(crop_data)
print("\nCrop data loaded and preprocessed.")



# Split data into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("\nData split into training and testing sets for classification.")

# Create and train the crop classification model
voting_clf = create_classification_model(X_train, y_train)

# Save the crop classification model
save_model(voting_clf, 'Models/voting_classifier_crop_model.joblib')

# Load and preprocess the yield prediction data
yield_data = load_yield_data("Data/Dataset.xlsx")  # Path to yield prediction data
# Features and target for yield prediction
X_yield, y_yield = preprocess_yield_data(yield_data)
print("\nYield prediction data loaded and preprocessed.")


# Split data into training and testing sets for yield prediction
X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(X_yield, y_yield, test_size=0.25, random_state=0)
print("\nData split into training and testing sets for yield prediction.")

# Create and train the yield prediction model
rf_regressor = create_regression_model(X_yield_train, y_yield_train)

# Save the yield prediction model
save_model(rf_regressor, 'Models/random_forest_yield_model.joblib')

print("\nModel creation and saving process completed successfully.")
