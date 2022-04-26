import numpy as np


def calculate_linear_kernel(a, b):
    linear_kernel = np.dot(a, b)
    return linear_kernel


def calculate_polynomial_kernel(a, b, constant, degree):
    polynomial_kernel = (np.dot(a, b) + constant) ** degree
    return polynomial_kernel


def calculate_gaussian_kernel(a, b, sigma):
    gaussian_kernel = np.exp(-(np.linalg.norm(a, b, axis=1)) ** 2 / (2 * sigma ** 2))
    return gaussian_kernel


class SVM:
    def __init__(self, kernel='gaussian', regularization_parameter_C=100):
        self.X_train = None
        self.y_train = None
        self.kernel = {'linear': lambda x, y: calculate_linear_kernel(x, y),
                       'polynomial_deg2': lambda x, y: calculate_polynomial_kernel(x, y, 0, 2),
                       'polynomial_deg3': lambda x, y: calculate_polynomial_kernel(x, y, 0, 3),
                       'gaussian': lambda x, y: calculate_gaussian_kernel(x, y, 1)}[kernel]
        self.regularization_parameter_C = regularization_parameter_C

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
