# data_preprocessing.py
''' 
# This module is responsible for loading and preprocessing the crop recommendation and yield prediction data.
# It includes functions to:
  - Load the crop recommendation data from a CSV file
  - Preprocess the crop data (handle missing values, rename columns)
  - Load the yield prediction data from an Excel file
  - Preprocess the yield data (select relevant features)
# The output is cleaned and structured data that is ready for training machine learning models.

'''
import pandas as pd

def load_crop_data(path: str):
    """Loads crop recommendation dataset."""
    # Load the dataset for crop recommendation
    crop_data = pd.read_csv(path)
    crop_data.rename(columns={'label': 'Crop'}, inplace=True)
    return crop_data

def preprocess_crop_data(crop_data):
    """Preprocesses the crop recommendation data."""
    crop_data = crop_data.dropna()  # Dropping rows with missing values

    # Features and Target variables for classification
    X = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features
    y = crop_data['Crop']  # Target variable
    return X, y

def load_yield_data(path: str):
    """Loads yield prediction dataset."""
    yield_df = pd.read_excel(path, sheet_name='Worksheet')
    return yield_df

def preprocess_yield_data(yield_df):
    """Preprocesses the yield prediction data."""
    # Selecting relevant features for yield prediction
    X_yield = yield_df[['N', 'P', 'K', 'pH', 'rainfall', 'temperature']]  # Features
    y_yield = yield_df['Yield_ton_per_hec']  # Target variable
    return X_yield, y_yield

'''
# Loading and preprocessing crop data
crop_data = load_crop_data('Data/crop_recommendation.csv') # --> Ensure correct path
crop_data = preprocess_crop_data(crop_data)

# Loading and preprocessing yield data
yield_df = load_yield_data('Data/Dataset.xlsx') # --> Ensure correct path
yield_df = preprocess_yield_data(yield_df)

# Indicating the completion of data preprocessing

print("\nData preprocessing completed successfully.")

'''
