# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

# Suppressing scikit-learn warning
warnings.filterwarnings(action='ignore', category=UserWarning)

# Load data from CSV file
data = pd.read_csv(r'C:\Users\vigne\OneDrive\Desktop\PRODIGY\MACHINE LEARNING\Task01\houseretail.csv')

# Splitting the data into features (X) and target variable (y)
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example prediction
example_house = np.array([[1800, 3, 2]])  # square footage = 1800, bedrooms = 3, bathrooms = 2
predicted_price = model.predict(example_house.reshape(1, -1))
print("Predicted price for the example house:", predicted_price)

# Visualization of regression model and prediction
plt.figure(figsize=(10, 6))

# Plotting actual vs. predicted prices for test set
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted Prices')
plt.plot(y_test, y_test, color='red', label='Perfect Fit')

plt.title('Linear Regression Model for House Prices Prediction')
plt.xlabel('Actual Prices ($)')
plt.ylabel('Predicted Prices ($)')
plt.legend()
plt.grid(True)
plt.show()
