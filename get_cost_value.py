import numpy as np


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


def convert_prob_into_class(Y_hat):
    Y_cop = np.copy(Y_hat)
    for i in range(len(Y_cop[0])):
        Y_cop[0, i] = 1 if Y_cop[0, i] > 0.5 else 0
    return Y_cop


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return 100 - np.mean(np.abs(Y_hat_ - Y)) * 100

# Y_hat = np.array([[0.6, 0.3, 0.1, 0.4]])
# Y = np.array([[1, 0, 0, 1]])
# accuracy = get_accuracy_value(Y_hat, Y)
# print(accuracy)
