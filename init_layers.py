import numpy as np
from nn_architecture import nn_architecture


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    params_values = {}

    for index, layer in enumerate(nn_architecture):
        layer_idx = index + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        params_values['W' + str(layer_idx)] = np.random.rand(layer_output_size, layer_input_size) * 0.01
        params_values['b' + str(layer_idx)] = np.random.rand(layer_output_size, 1) * 0.01
        # print(params_values['W' + str(layer_idx)].shape)
        # print(params_values['b' + str(layer_idx)].shape)

    return params_values


params_values = init_layers(nn_architecture, 3)
