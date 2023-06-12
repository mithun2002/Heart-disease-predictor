import pandas as pd

# Create a dictionary with the data
data = {
    'weight': [70, 65, 80, 75, 72],
    'chol': [200, 190, 220, 205, 198],
    'bp': [120, 130, 110, 140, 125]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv('heart_disease_dataset.csv', index=False)