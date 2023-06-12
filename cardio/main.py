import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the dataset
heart_data = pd.read_csv('heart_disease_dataset.csv')

# Separate features (X) and target variable (y)
X = heart_data[['weight', 'chol', 'bp']]
y = heart_data['target']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)
