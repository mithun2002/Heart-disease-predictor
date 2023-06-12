import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
heart_data = pd.read_csv('heartdata.csv')

# Separate features (X) and target variable (y)
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Get input from the user
Bp = int(input("Enter age: "))
Gender = int(input("Enter sex (0 for female, 1 for male): "))
Weight = int(input("Enter chest pain type (0-3): "))


# Create a DataFrame from the user input
user_data = pd.DataFrame({
    'bp': [Bp],
    'gender': [Gender],
    'weight': [Weight]
})

# Standardize the user input
user_data_scaled = scaler.transform(user_data)

# Make prediction on the user input
prediction = model.predict(user_data_scaled)

# Interpret the prediction
if prediction[0] == 0:
    print("The model predicts that the person Edoes not have cardiovascular disease.")
else:
    print("The model predicts that the person has cardiovascular disease.")
