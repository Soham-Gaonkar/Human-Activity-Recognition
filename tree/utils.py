"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    for column in X.columns:
        if (not check_ifreal(X[column])) & (len(X[column].unique()) > 2):
            X = pd.concat([X, pd.get_dummies(X[column], prefix=column)], axis=1)
            X.drop(column, axis=1, inplace=True)
    return X



def check_ifreal(y: pd.Series, real_distinct_threshold: int = 15) -> bool:
    """
    Function to check if the given series has real or discrete values

    Returns True if the series has real (continuous) values, False otherwise (discrete).

    """

    if pd.api.types.is_categorical_dtype(y):
        return False
    if pd.api.types.is_bool_dtype(y):
        return False
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_integer_dtype(y):
        return len(y.unique()) > real_distinct_threshold
    if pd.api.types.is_string_dtype(y):
        return False
    return False



def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy

    entropy = -sum(p_i * log2(p_i))
    """

    value_counts = Y.value_counts()
    total_count = Y.size
    prob = value_counts / total_count
    entropy_value = -np.sum(prob * np.log2(prob + 1e-10)) # Adding a small value to avoid log(0)
    return entropy_value



def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index

    gini_index = 1 - sum(p_i^2)
    """

    value_counts = Y.value_counts()
    total_count = Y.size
    probs = value_counts / total_count
    gini_index_value = 1 - np.sum(probs ** 2)
    return gini_index_value



def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error (mse)

    mse = sum((y_i - y)^2) / n
    """

    Y_mean = Y.mean()
    mse_value = np.sum((Y - Y_mean) ** 2) / Y.size
    return mse_value



def check_criteria(Y:pd.Series, criterion: str):
    """
    Function to check if the criterion is valid
    """

    if criterion == "information_gain":
        if check_ifreal(Y):
            my_criterion = 'mse'
        else:
            my_criterion = 'entropy'
    elif criterion == "gini_index":
        my_criterion = 'gini_index'
    else:
        raise ValueError("Criterion must be 'information_gain' or 'gini_index'.")
    
    criterion_funcs_map = {
        'entropy': entropy,
        'gini_index': gini_index,
        'mse': mse
    }
    criterion_func = criterion_funcs_map[my_criterion]

    return my_criterion, criterion_func



def find_optimal_threshold(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to find the optimal threshold for a real feature

    Returns the threshold value for best split in a given real feature
    """

    my_criterion, criterion_func = check_criteria(Y, criterion)

    sorted_attr = attr.sort_values()
    # Find the split points by taking the average of consecutive values (midpoints)
    if sorted_attr.size == 1:
        return None
    elif sorted_attr.size == 2:
        return (sorted_attr.sum()) / 2
    split_points = (sorted_attr[:-1] + sorted_attr[1:]) / 2

    best_threshold = None
    best_gain = -np.inf

    for threshold in split_points:
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]

        if Y_left.empty or Y_right.empty:
            continue

        total_criterion = Y_left.size / Y.size * criterion_func(Y_left) + Y_right.size / Y.size * criterion_func(Y_right)
        information_gain_value = criterion_func(Y) - total_criterion

        if information_gain_value > best_gain:
            best_threshold = threshold
            best_gain = information_gain_value

    return best_threshold



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)

    information_gain = criterion(Y) - sum((Y_i.size / Y.size) * criterion(Y_i))
    """

    my_criterion, criterion_func = check_criteria(Y, criterion)

    # If the attribute is real, find the split points and calculate the information gain for each split point
    if check_ifreal(attr):
        threshold = find_optimal_threshold(Y, attr, criterion)
        if threshold is None:
            return 0  # No valid threshold found
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]
        information_gain_value = criterion_func(Y) - (Y_left.size / Y.size * criterion_func(Y_left) + Y_right.size / Y.size * criterion_func(Y_right))
        return information_gain_value
    
    # If the attribute is discrete, calculate the information gain for each unique value of the attribute
    total_criterion = 0
    for value in attr.unique():
        Y_i = Y[attr == value]
        total_criterion += (Y_i.size / Y.size) * criterion_func(Y_i)
    information_gain_value = criterion_func(Y) - total_criterion
    return information_gain_value



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion: str) -> str:
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    best_feature = None
    best_gain = -np.inf
    for feature in features:
        gain = information_gain(y, X[feature], criterion)
        if gain > best_gain:
            best_feature = feature
            best_gain = gain
    return best_feature



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if check_ifreal(X[attribute]):
        X_left = X[X[attribute] <= value]
        X_right = X[X[attribute] > value]
    else:
        X_left = X[X[attribute] == value]
        X_right = X[X[attribute] != value]
    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]
    return X_left, y_left, X_right, y_right
