# California Housing Price Prediction Using Linear Regression

This repository contains a Python implementation of a linear regression model to predict California housing prices based on selected features from the California Housing dataset.

## Dataset

The dataset used is the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) provided by scikit-learn, which includes various features related to housing in California and the median house values.

## Features Used

- `MedInc` (Median Income)
- `AveRooms` (Average Rooms per Household)

You can modify the features in the code to experiment with different predictors.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/California-Housing-Linear-Regression.git
   cd California-Housing-Linear-Regression
# Housing-Price-Prediction-LinearModel

pip install -r requirements.txt

python housing_regression.py

The script will:

Load and preprocess the dataset

Split it into training and test sets

Train a linear regression model

Output evaluation metrics (MSE and R² score)

Plot results if a single feature is used

Evaluation Metrics

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.

R² Score: Indicates the proportion of variance in the target variable explained by the features.

Requirements

Python 3.7+

numpy

pandas

matplotlib
