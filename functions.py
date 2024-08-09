"""
This module provides various utility functions for neural networks, 
including activation functions, loss functions, and data loading functions.
"""

import os
import pickle
from typing import Tuple

import numpy as np
from mnist import MNIST


def softmax(x: np.ndarray):
    """
    The softmax activation function.

    Parameters:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Softmax-transformed array.
    """
    # Subtracting the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def relu(x: np.ndarray):
    """
    The Rectified Linear Unit (ReLU) activation function.

    Parameters:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: ReLU-transformed array.
    """
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray):
    """
    Derivative of the ReLU activation function.

    Parameters:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Array of derivatives.
    """
    return np.where(x > 0, 1, 0)

def categorical_cross_entropy(predicted: np.ndarray, actual: np.ndarray):
    """
    A loss function suited for categorical data.

    Parameters:
    predicted (np.ndarray): Predicted probabilities.
    actual (np.ndarray): Actual labels.

    Returns:
    float: Categorical cross-entropy loss.
    """
    epsilon = 1e-12  # To avoid log(0)
    predicted = np.clip(predicted, epsilon, 1. - epsilon)
    return -np.sum(actual * np.log(predicted))

def load_mnist_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset from the specified directory.

    Parameters:
    data_dir (str): Directory containing the MNIST data files.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - X_train (np.ndarray): Training data features.
        - y_train (np.ndarray): Training data labels.
        - X_test (np.ndarray): Testing data features.
        - y_test (np.ndarray): Testing data labels.

    Example:
    >>> X_train, y_train, X_test, y_test = load_mnist_data('/path/to/mnist/data')
    """
    mndata = MNIST(data_dir)

    X_train, y_train = mndata.load_training()
    X_test, y_test = mndata.load_testing()

    # Convert lists to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

def load_model(filename: str, dir: str = "models") -> object:
    """
    Loads a model from a file.

    Parameters:
    filename (str): Name of the file containing the model.
    dir (str): Directory containing the model file. By default, it is "models".

    Returns:
    object: The loaded model.

    Example:
    >>> model = load_model("mnist_model", "models")
    """

    filepath = os.path.join(dir, f"{filename}.pkl")

    with open(filepath, 'rb') as file:
        model = pickle.load(file)

    return model
