from single_layer_backward_propagation import *


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    # print("Y in full_backward_propation: ")
    # print(Y)
    # print("Y_hat in full_backward_propation: ")
    # print(Y_hat)
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    dA_curr = dA_prev
    # print("dA_curr in full_backward_progation: ")
    # print(dA_curr)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        # print(activ_function_curr)

        A_prev = memory["A" + str(layer_idx_prev)]
        # print("A_prev in full_backward_propagation: ")
        # print(A_prev)
        Z_curr = memory["Z" + str(layer_idx_curr)]
        # print("Z_curr in full_backward_propagation: ")
        # print(Z_curr)
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values
