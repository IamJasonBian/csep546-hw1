"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np
import pandas as pd
import pdb
from sklearn.preprocessing import PolynomialFeatures

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=4)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """
        Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        self.weight = None 
        self.X_mean = None
        self.X_std = None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        poly = PolynomialFeatures(degree=degree)
        X = poly.fit_transform(X.reshape(-1,1))
      
        return(X[:,1:])

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        n = len(X)

        # add additional polynomial features
        X = self.polyfeatures(X, self.degree)

        # standardize all features
        self.X_mean = np.mean(X, axis = 0)
        self.X_std = np.std(X, axis = 0)
        X = (X-self.X_mean)/self.X_std
    
        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        n, d = X_.shape

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(d)
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """

        n = len(X)
        X = self.polyfeatures(X, self.degree)

        # standardize all features
        X = (X-self.X_mean)/self.X_std

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        print(X_)
        # predict
        return X_.dot(self.weight).reshape((-1,1))

@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)
    
    

    Returns:
        float: mean squared error between a and b.
    """

    difference_array = np.subtract(a, b)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()

    return mse


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    raise NotImplementedError("Your Code Goes Here")
