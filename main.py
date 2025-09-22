# %%




# %%


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# %%
# Load the California housing dataset
housing = fetch_california_housing()

# Create DataFrame for features and target
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target


# %%
# Select specific features for linear regression
features = ['MedInc', 'AveRooms']  # You can change this to only one feature for plotting
X_selected = X[features]


# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)


# %%
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# %%
# Predict on test data
y_pred = model.predict(X_test)


# %%
# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# %%
# Only plot if a single feature is selected
if X_selected.shape[1] == 1:
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel(features[0])
    plt.ylabel("Median House Value")
    plt.title("Linear Regression - Housing Prices")
    plt.legend()
    plt.show()



