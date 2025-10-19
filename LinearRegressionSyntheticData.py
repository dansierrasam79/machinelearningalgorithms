from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
import numpy as np

# 1. Generate sample data (multiple features for a more robust example)
# X represents the inputs, y represents the actual outputs
X, y = make_regression(n_samples=100, n_features=3, noise=20, random_state=42)

# 2. Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions used the X_test dataset 
y_pred = model.predict(X_test)

# 5. Evaluate the model
# mse computes the magnitude of error between data used for testing and the predicted values
# the lower the mse, the better
mse = mean_squared_error(y_test, y_pred)
# goodness-of-fit metric that determines how much variation in the 
# independent (Input) variable influences the dependent (target) variable
# ranges between 0 and 1, the higher r2 score, the better
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# You can also access the learned coefficients and intercept
print(f"Coefficients (weights): {model.coef_}")
print(f"Intercept (bias): {model.intercept_:.2f}")
