import numpy as np
from sklearn.linear_model import LinearRegression

# Example dataset: [size_in_sqft, number_of_rooms] and target house prices
X = np.array([
    [800, 2],
    [1200, 3],
    [1500, 4],
    [1000, 3],
    [2000, 5]
])
y = np.array([350000, 500000, 680000, 420000, 850000])

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict the price of a new house (e.g., 1750 sqft, 4 rooms)
new_house = np.array([[1750, 4]])
predicted_price = model.predict(new_house)

print(f"Predicted price for a 1750 sq. ft, 4-room house: ${predicted_price[0]:,.2f}")
