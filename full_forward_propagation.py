from single_layer_forward_propagation import *


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for index, layer in enumerate(nn_architecture):
        layer_index = index + 1
        A_prev = A_curr
        activation_func_curr = layer['activation']
        W_curr = params_values['W' + str(layer_index)]
        b_curr = params_values['b' + str(layer_index)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_func_curr)
        # print(layer['activation'])
        # print("A_curr in full_forward_propagation: ")
        # print(A_curr)
        memory['A' + str(index)] = A_prev
        memory['Z' + str(layer_index)] = Z_curr

    return A_curr, memory
