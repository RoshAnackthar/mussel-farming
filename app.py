import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset, handling commas in numeric columns
data = pd.read_csv('/content/csp.csv', thousands=',') # Handle commas

# Define features and target
features =[
    'Year', 'Total net production volume (kg)',
    'Expected price (Euro/Kg)', 'Revenue (Euro)', 'Yearly Fixed cost', 'Variable cost',
    'Cash Flow'
] # Removed extra brackets
target = 'Net Profit'

# Separate features and target
X = data[features] # Use features list directly
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate model accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot predictions against actual values
plt.figure(figsize=(10, 6))
plt.scatter(data['Year'], y, color='blue', label='Actual Net Profit')
plt.scatter(data['Year'].iloc[X_test.index], y_pred, color='red', label='Predicted Net Profit')
plt.xlabel('Year')
plt.ylabel('Net Profit')
plt.title('Net Profit Prediction')
plt.legend()
plt.show()
