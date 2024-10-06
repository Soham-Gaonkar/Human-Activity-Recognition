from typing import Union
import pandas as pd

def validate_inputs(y_hat, y):
    """
    The following assert checks if sizes of y_hat and y are equal.
    assert y_hat.size == y.size
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
        
    # Ensure both y_hat and y are pandas Series
    assert isinstance(y_hat, pd.Series), "y_hat must be a pandas Series"
    assert isinstance(y, pd.Series), "y must be a pandas Series"

    # Ensure both y_hat and y have the same length
    assert y_hat.size == y.size, "y_hat and y must have the same size"

    # Ensure neither y_hat nor y is empty
    assert y_hat.size > 0, "y_hat cannot be empty"
    assert y.size > 0, "y cannot be empty"

    # Reset the index to ensure alignment by values
    y_hat = y_hat.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return y_hat, y


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    TP: True Positives
    TN: True Negatives
    FP: False Positives
    FN: False Negatives
    """

    y_hat, y = validate_inputs(y_hat, y)
    accuracy_value = (y_hat == y).sum() / y.size
    return accuracy_value


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision

    Precision = TP / (TP + FP)
    TP: True Positives
    FP: False Positives
    """

    y_hat, y = validate_inputs(y_hat, y)
    TP = ((y_hat == cls) & (y == cls)).sum()
    FP = ((y_hat == cls) & (y != cls)).sum()
    precision_value = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precision_value


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall

    Recall = TP / (TP + FN)
    TP: True Positives
    FN: False Negatives
    """

    y_hat, y = validate_inputs(y_hat, y)
    TP = ((y_hat == cls) & (y == cls)).sum()
    FN = ((y_hat != cls) & (y == cls)).sum()
    recall_value = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recall_value
    

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)

    RMSE = sqrt(1/n * sum(y_hat - y)^2)
    """
    
    y_hat, y = validate_inputs(y_hat, y)
    rmse_value = ((y_hat - y) ** 2).sum() / y.size
    rmse_value = rmse_value ** 0.5
    return rmse_value


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)

    MAE = 1/n * sum(|y_hat - y|)
    """
    
    y_hat, y = validate_inputs(y_hat, y)
    mae_value = (y_hat - y).abs().sum() / y.size
    return mae_value    
