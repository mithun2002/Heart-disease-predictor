import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
heart_data = pd.read_csv('heart_disease_dataset.csv')

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
age = int(input("Enter age: "))
sex = int(input("Enter sex (0 for female, 1 for male): "))
cp = int(input("Enter chest pain type (0-3): "))
trestbps = int(input("Enter resting blood pressure: "))
chol = int(input("Enter cholesterol level: "))
fbs = int(input("Enter fasting blood sugar (> 120 mg/dl) (0 for False, 1 for True): "))
restecg = int(input("Enter resting electrocardiographic results (0-2): "))
thalach = int(input("Enter maximum heart rate achieved: "))
exang = int(input("Enter exercise-induced angina (0 for No, 1 for Yes): "))
oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
slope = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
ca = int(input("Enter number of major vessels colored by fluoroscopy (0-3): "))
thal = int(input("Enter thalassemia (0-3): "))

# Create a DataFrame from the user input
user_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Standardize the user input
user_data_scaled = scaler.transform(user_data)

# Make prediction on the user input
prediction = model.predict(user_data_scaled)

# Interpret the prediction
if prediction[0] == 0:
    print("The model predicts that the person does not have cardiovascular disease.")
else:
    print("The model predicts that the person has cardiovascular disease.")
