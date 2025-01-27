# Crop Recommendation and Yield Prediction

## Overview
This project uses machine learning models to predict the most suitable crop based on various environmental and soil factors (e.g., nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall) and to estimate the expected yield for a given crop. It provides:
1. **Crop Recommendation**: Predicts the most suitable crop for a set of environmental and soil conditions.
2. **Yield Prediction**: Estimates the expected yield of a crop (in tons per hectare) based on the same set of conditions.

The project uses machine learning models like **Random Forest**, **K-Nearest Neighbors (KNN)**, **Gradient Boosting**, and **Voting Classifier** for classification, and **RandomForestRegressor** for yield prediction.

## Project Structure

Crop_Prediction/
    ├── Code/
    │   ├── app.py                   # Main Flask application to run the web app
    │   ├── data_preprocessing.py     # Data loading and preprocessing script
    │   ├── model_creation.py         # Script to create and train models
    │   ├── model_usage.py            # Script to load models and make predictions
    │   ├── prediction.py            # Script to trigger model training
    │   ├── /static/
    │   │   └── /css/
    │   │       └── style.css        # Custom CSS file for styling the web interface
    │   └── /templates/
    │       ├── index.html           # Input form for users to provide data
    │       └── result.html          # Page to display prediction results
    ├── Data/
    │   ├── Crop_recommendation.csv  # Dataset for crop recommendation
    │   └── Dataset.xlsx             # Dataset for yield prediction
    ├── requirements.txt             # List of required libraries to run the project
    └── README.md                    # Project overview and instructions

### Project Components

1. **Model Creation (Four Key Files)**:
   The machine learning models for crop recommendation and yield prediction are created and trained in the following four files:

   - **data_preprocessing.py**:
     - This file handles loading and preprocessing of the crop recommendation and yield prediction datasets. It includes data cleaning, handling missing values, feature selection, and scaling.
     - Functions: `load_crop_data()`, `preprocess_crop_data()`, `load_yield_data()`, `preprocess_yield_data()`
   
   - **model_creation.py**:
     - This script defines the model creation and training logic.
     - **Crop Recommendation**: A **Voting Classifier** is used, which is an ensemble of three classifiers: **Random Forest**, **K-Nearest Neighbors (KNN)**, and **Gradient Boosting**.
     - **Yield Prediction**: A **Random Forest Regressor** is used for predicting the yield.
     - The models are then saved using **joblib** for future use.
     - Functions: `create_classification_model()`, `create_regression_model()`, `save_model()`

   - **model_usage.py**:
     - This file is responsible for loading the trained models and making predictions based on user input.
     - It also evaluates the models using metrics like accuracy for classification and mean squared error for regression.
     - Functions: `predict_crop_and_yield()`
   
   - **train_model.py** (or integrated into **model_creation.py**):
     - Used to trigger the process of training and saving the models.
     - This file is responsible for initiating the training process, splitting the data into training and testing sets, and saving the models for deployment.

2. **Interface Part (Flask Web Application)**:
   - **app.py**: 
     - This file defines the main Flask application. It loads the pre-trained models, handles user input via forms, and renders the predictions (crop and yield) on a new page.
     - The app has two main routes:
       - `/`: Displays the input form where users can enter soil and weather conditions.
       - `/predict`: Handles the form submission, makes predictions using the trained models, and displays the results on a new page.
   
   - **Templates**:
     - `index.html`: This file defines the user interface for inputting soil and weather conditions. It includes various form fields such as nitrogen, phosphorous, potassium content, temperature, humidity, pH, and rainfall.
     - `result.html`: This file displays the results of the prediction, including the recommended crop and estimated yield.
   
   - **Static files**:
     - `style.css`: Defines custom styles for the web application, including color schemes, fonts, and layout adjustments to make the app responsive and visually appealing.
     - You can further improve the UI by customizing the CSS to match your desired look and feel.


## Dependencies

This project requires the following libraries:

- **numpy**: For numerical computations and array manipulations.
- **pandas**: For data manipulation and processing.
- **matplotlib**: For data visualization (charts and graphs).
- **seaborn**: For advanced visualizations, especially heatmaps.
- **scikit-learn**: For machine learning algorithms and model evaluation.
- **xgboost**: For advanced regression models, specifically for yield prediction.
- **openpyxl**: For reading/writing Excel files (`.xlsx` format).
- **joblib** (Optional): For saving and loading trained models to avoid retraining.
- **Flask**: For the web framework

### To Install Dependencies:
Use the following command to install all required libraries via `pip`:
```bash
pip install -r requirements.txt

```

## Dataset
1. Crop Recommendation Dataset (Crop_recommendation.csv)
This dataset contains information about soil and environmental factors along with the corresponding crop recommendations. The columns in this dataset include:

- N (Nitrogen content)
- P (Phosphorous content)
- K (Potassium content)
- temperature (Temperature in °C)
- humidity (Humidity in %)
- ph (pH value of soil)
- rainfall (Rainfall in mm)
- label (Crop type)

2. Yield Prediction Dataset (Dataset.xlsx)
This dataset contains soil and environmental factors, and the corresponding crop yield (tons per hectare). The columns in this dataset include:

- N (Nitrogen content)
- P (Phosphorous content)
- K (Potassium content)
- pH (pH value of soil)
- rainfall (Rainfall in mm)
- temperature (Temperature in °C)
- Yield_ton_per_hec (Crop yield in tons per hectare)

Note: Ensure that the dataset files are located in the Data/ directory and that the paths in the code are correctly updated.

## Features
- Crop Recommendation: Predicts the best crop to grow based on soil and weather conditions.
- Yield Prediction: Predicts the expected yield of a specific crop based on the same factors.
- Responsive Web Interface: Built with Flask and styled with custom CSS for a clean and responsive design.
- Real-time Prediction: Users can enter their soil and weather data, and the model will predict the crop and yield in real-time.

## How to Run the Project

# Install Dependencies:
Make sure all dependencies are installed by running:
```bash
pip install -r requirements.txt
```
# Run the Script: 
Run the main script to get crop recommendations and yield predictions based on your inputs.
```bash
python Code/app.py
```

# User Input: 
The program will prompt you to input the following values for both crop recommendation and yield prediction:

- Nitrogen content (N)
- Phosphorous content (P)
- Potassium content (K)
- Temperature (°C)
- Humidity (%)
- pH value of soil
- Rainfall (mm)

After providing the inputs, the system will:

- Recommend a crop based on the input data.
- Predict the yield of the recommended crop (in tons per hectare).