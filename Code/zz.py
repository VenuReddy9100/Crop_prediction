# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from xgboost import XGBRegressor

# Load the dataset for crop recommendation
crop_data = pd.read_csv("Data/Crop_recommendation.csv")  # ---> Update path
crop_data.rename(columns={'label': 'Crop'}, inplace=True) # Renaming the lable column as Crop

# Checking for missing values and dropping them if any
crop_data = crop_data.dropna()

# Features and Target variables for classification
X = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] 
y = crop_data['Crop']                                                      

# Splitting data into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

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

# Predict on test data for classification
y_pred = voting_clf.predict(X_test)

# Evaluate the classification model
accuracy = accuracy_score(y_test, y_pred)
print("Hybrid Model Accuracy (Classification):", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix for classification
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Classification)')
plt.show()

# Load the dataset for yield prediction
# Ensure you replace the file path with the correct path to your Dataset.xlsx file
yield_df = pd.read_excel("Data/Dataset.xlsx", sheet_name='Worksheet')   # ---> Update Path

# Selecting relevant features for yield prediction
X_yield = yield_df[['N', 'P', 'K', 'pH', 'rainfall', 'temperature']]
y_yield = yield_df['Yield_ton_per_hec']

# Splitting data into training and testing sets for yield prediction
X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(X_yield, y_yield, test_size=0.25, random_state=0)

# Define and train the model for yield prediction
rf_regressor = RandomForestRegressor(random_state=1)
rf_regressor.fit(X_yield_train, y_yield_train)



# Function to take user input, predict the crop, and estimate yield
def predict_crop_and_yield():
    print("\nEnter the following details for crop recommendation and yield prediction:")
    N = float(input("Nitrogen content (N): "))
    P = float(input("Phosphorous content (P): "))
    K = float(input("Potassium content (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH value of soil: "))
    rainfall = float(input("Rainfall (mm): "))

    # Creating a DataFrame for the input
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Making a prediction for the crop
    crop_prediction = voting_clf.predict(input_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
    print(f"\nRecommended Crop: {crop_prediction[0]}")

    # Predicting the yield for the input
    input_yield_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'pH': [ph],
        'rainfall': [rainfall],
        'temperature': [temperature]
    })
    yield_prediction = rf_regressor.predict(input_yield_data)
    print(f"Expected Yield: {yield_prediction[0]:.2f} tons/ha")

# Take user input and predict the crop and yield
predict_crop_and_yield()


