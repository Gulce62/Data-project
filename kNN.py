import numpy as np
from statistics import mode


def calculate_euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    euclidian_distance = np.linalg.norm(a - b)
    return euclidian_distance


def calculate_manhattan_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    size = len(a)
    man_distance = 0
    for i in range(size):
        man_distance += abs(a[i]-b[i])
    return man_distance


def calculate_cosine_distance(a, b):
    cos_distance = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_distance


class KNN():
    def __init__(self, k=3, distance_type=1):
        self.X_train = None
        self.y_train = None
        self.k = k
        self.distance_type = distance_type

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def emotion_predict(self, X_test):
        a_distance = np.array([])
        for X in self.X_train:
            if self.distance_type == 1:
                distance = calculate_euclidean_distance(X_test, X)
            elif self.distance_type == 2:
                distance = calculate_manhattan_distance(X_test, X)
            else:
                distance = calculate_cosine_distance(X_test, X)
            a_distance = np.append(a_distance, distance)
        knn_indices = a_distance.argsort()[:self.k]
        a_knn_neighbors = np.array([])
        for index in knn_indices:
            a_knn_neighbors = np.append(a_knn_neighbors, self.y_train[index])
        return mode(a_knn_neighbors)

    def all_prediction(self, X_train):
        a_emotions_predictions = np.array([])
        for X in X_train:
            prediction = self.emotion_predict(X)
            a_emotions_predictions = np.append(a_emotions_predictions, prediction)
        return a_emotions_predictions


