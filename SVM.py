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
    def __init__(self, kernel='gaussian', regularization_parameter_C=100, max_iterations=60):
        self.X_train = None
        self.y_train = None
        self.matrix_K = None
        self.support_vector = None
        self.alpha_variables = None
        self.bias = None
        self.kernel = {'linear': lambda x, y: calculate_linear_kernel(x, y),
                       'polynomial_deg2': lambda x, y: calculate_polynomial_kernel(x, y, 0, 2),
                       'polynomial_deg3': lambda x, y: calculate_polynomial_kernel(x, y, 0, 3),
                       'gaussian': lambda x, y: calculate_gaussian_kernel(x, y, 1)}[kernel]
        self.regularization_parameter_C = regularization_parameter_C
        self.max_iterations = max_iterations

    def fit(self, X_train, y_train):
        self.X_train = X_train.copy()
        self.y_train = y_train * 2 - 1
        kernel_matrix = self.kernel(self.X_train, self.X_train)
        y_matrix = self.y_train * self.y_train[:, np.newaxis]
        threshold = 1e-15
        self.matrix_K = kernel_matrix * y_matrix
        self.alpha_variables = self.find_alphas(y_train, self.max_iterations, threshold)
        self.bias = self.get_bias(threshold)

    def find_alphas(self, y_train, max_iterations, threshold):
        (row, column) = y_train.shape
        alpha_variables = np.zeros((row, column), dtype=float)
        while max_iterations != 0:
            length = len(alpha_variables)
            for index_m in range(length):
                index_l = np.random.randint(0, length)
                main_index = [index_m, index_l]
                q_parameter = self.matrix_K[[[index_m, index_m], [index_l, index_l]], [main_index, [index_l, index_m]]]
                v0_parameter = alpha_variables[main_index]
                m_l_indices = alpha_variables * self.matrix_K[main_index]
                k0_parameter = 1 - np.sum(m_l_indices, axis=1)
                u_value = np.array(-self.y_train[index_l], self.y_train[index_m])
                maximum_t = np.dot(k0_parameter, u_value) / (np.dot(np.dot(q_parameter, u_value), u_value) + threshold)
                v_function = v0_parameter + u_value * maximum_t
                if maximum_t[0] or maximum_t[1] > self.regularization_parameter_C:
                    t_res = (np.clip(v_function, 0, self.regularization_parameter_C))[1] / u_value[1]
                    new_v_function = v0_parameter + u_value * t_res
                    t_res = (np.clip(new_v_function, 0, self.regularization_parameter_C))[0] / u_value[0]
                    v_function = v0_parameter + u_value * t_res
                alpha_variables[main_index] = v_function
        return alpha_variables

    def get_bias(self, threshold):
        non_zero_indices = np.nonzero(self.alpha_variables > threshold)
        summation = np.sum(self.matrix_K[non_zero_indices] * self.alpha_variables, axis=1)
        bias = np.mean((1.0 - summation) * self.y_train[non_zero_indices])
        return bias

    def emotion_predict(self, X_train):
        new_kernel_matrix = self.kernel(X_train, self.X_train)
        new_y_matrix = self.y_train * self.alpha_variables
        support_vector = np.sum(new_kernel_matrix * new_y_matrix, axis=1) + self.bias
        sign_prediction = (np.sign(support_vector + 1)) // 2
        return sign_prediction
