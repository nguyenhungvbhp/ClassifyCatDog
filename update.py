def update(params_values, grads_values, nn_architecture, learning_rate):
    # print(grads_values)
    for layer_idx, layer in enumerate(nn_architecture):
        index = layer_idx + 1
        # print(grads_values["dW" + str(index)])
        params_values["W" + str(index)] -= learning_rate * grads_values["dW" + str(index)]
        params_values["b" + str(index)] -= learning_rate * grads_values["db" + str(index)]

    return params_values;
