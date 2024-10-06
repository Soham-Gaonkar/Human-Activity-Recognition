import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features
y = auto_mpg.data.targets 
  
# metadata 
print(auto_mpg.metadata) 
  
# variable information 
print(auto_mpg.variables)

# join X and y to check for null values
data = pd.concat([X, y], axis=1)
print("Shape of extracted data: ", data.shape)

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
# data = pd.read_csv(url, delim_whitespace=True, header=None,
#                  names=["mpg", "cylinders", "displacement", "horsepower", "weight",
#                         "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn



# Data Cleaning/Preprocessing
data.replace('?', np.nan, inplace=True)
print("\nNumber of NaN/Null values in training data:", data.isnull().sum().sum())
if data.isnull().sum().sum() > 0:
    data.dropna(inplace=True)
print("Number of duplicated samples in training data: ", data.duplicated().sum())
if data.duplicated().sum() > 0:
    data.drop_duplicates(inplace=True)

print("Shape of data after cleaning: ", data.shape)


# separate X and y
X = data.drop('mpg', axis=1)
y = data['mpg']
print("\nX shape:", X.shape)
print("y shape:", y.shape)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nX_train size: ", X_train.shape)
print("y_train size: ", y_train.shape)
print("X_test size: ", X_test.shape)
print("y_test size: ", y_test.shape)


# Our custom decision tree implementation
my_dt = DecisionTree(criterion="information_gain", max_depth=5)
my_dt.fit(X_train, y_train)

y_train_pred = my_dt.predict(X_train)
y_test_pred = my_dt.predict(X_test)

train_rmse = rmse(y_train_pred, y_train)
train_mae = mae(y_train_pred, y_train)

print("\nTrain Metrics (Custom):")
print(f"    Root Mean Squared Error: {train_rmse:.4f}")
print(f"    Mean Absolute Error: {train_mae:.4f}")

test_rmse = rmse(y_test_pred, y_test)
test_mae = mae(y_test_pred, y_test)

print("\nTest Metrics (Custom):")
print(f"    Root Mean Squared Error: {test_rmse:.4f}")
print(f"    Mean Absolute Error: {test_mae:.4f}")



# Scikit-learn's Decision Tree
sklearn_model = DecisionTreeRegressor(max_depth=5)
sklearn_model.fit(X_train, y_train)

y_train_pred_sklearn = sklearn_model.predict(X_train)
y_test_pred_sklearn = sklearn_model.predict(X_test)

train_rmse_sklearn = rmse(pd.Series(y_train_pred_sklearn), y_train)
train_mae_sklearn = mae(pd.Series(y_train_pred_sklearn), y_train)

print("\nTrain Metrics (Sklearn):")
print(f"    Root Mean Squared Error: {train_rmse_sklearn:.4f}")
print(f"    Mean Absolute Error: {train_mae_sklearn:.4f}")

test_rmse_sklearn = rmse(pd.Series(y_test_pred_sklearn), y_test)
test_mae_sklearn = mae(pd.Series(y_test_pred_sklearn), y_test)

print("\nTest Metrics (Sklearn):")
print(f"    Root Mean Squared Error: {test_rmse_sklearn:.4f}")
print(f"    Mean Absolute Error: {test_mae_sklearn:.4f}")



# Performance Comparison
print("\nPerformance Comparison:\n")
print(f"Our Decision Tree - Train RMSE: {train_rmse:.4f}")
print(f"Our Decision Tree - Test RMSE: {test_rmse:.4f}")
print(f"Scikit-Learn Decision Tree - Train RMSE: {train_rmse_sklearn:.4f}")
print(f"Scikit-Learn Decision Tree - Test RMSE: {test_rmse_sklearn:.4f}")