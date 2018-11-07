import numpy as np
from full_backward_propagation import full_backward_propagation
from full_forward_propagation import full_forward_propagation
from init_layers import init_layers
from get_cost_value import *
from update import update
from load_dataset import standardize_data
from nn_architecture import nn_architecture


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)
        # Y_hat_copy = np.copy(Y_hat)
        # print("Y_hat in train: ")
        # print(Y_hat)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)

        # print("Y_hat in train: ")
        # print(Y_hat)
        # print("Y_hat_copy in train: ")
        # print(Y_hat_copy)
        grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
        # print("grads_values: ")
        # print(grads_values)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
        print(params_values)
        print('Cost after iteration ' + str(i) + ' : ' + str(accuracy))
    return params_values, cost_history, accuracy_history


train_set_x, train_set_y, test_set_x, test_set_y, classes = standardize_data()

train(train_set_x, train_set_y, nn_architecture, 10, 0.5)
