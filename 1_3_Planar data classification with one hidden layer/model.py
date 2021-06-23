import numpy as np
from planar_utils import *


def layer_sizes(X, Y):
    """
    parameters:
    X, input data (shape dim, num)
    Y, (shape 1, num)

    return:
    n_x: number of nodes in this input layer
    n_h: number of units in this hidden layer
    n_y: number of nodes in this output layer

    """

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """

    :param n_x:
    :param n_h:
    :param n_y:


    :return:
        (Return value has two array of W,b, since that this nn only has two layer(except for input layer))

        parameters - one dictionary:
        W1:  the hidden layer W , shape(n_h, n_x)
        b1:  the hidder layer b , shape(n_h, 1)
        W2:  the output layer W , shape(n_y, n_h)
        b2:  the output layer b , shape(n_y, 1)
    """

    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2

    }

    return parameters


def forward_propagation(X, parameters):
    """

    :param X: input data
    :param parameters: parameters of hidden layer and output layer


    :return:
            A2: the final output value
            cache: one dictionary including Z1, A1, Z2, A2

    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1  # shape(n_h, num)
    A1 = np.tanh(Z1)  # shape(n_h, num)
    Z2 = np.dot(W2, A1) + b2  # shape(n_Y, num)
    A2 = sigmoid(Z2)  # shape(n_Y, num)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def backward_propagation(paramters, cache, X, Y):
    '''

    :param X:
    :param Y:
    :return:
        grads(W2, b2, W1, b1)
    '''

    m = Y.shape[1]
    W2 = paramters["W2"]  # shape n_y, n_h

    A1 = cache["A1"]  # shape n_h, num
    A2 = cache["A2"]

    dZ2 = A2 - Y  # shape n_y, num
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))  # shape n_h, num
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def compute_cost(A2, Y, parameters):
    '''

    :param A2:  the predict Y (shape n_Y, num)
    :param Y:   the ground truth (shape 1, num)
    :param parameters:


    :return:
        cost:
    '''

    m = Y.shape[1]  # the number of samplers

    # (shape n_Y, num), np.multiply is the inner product operations, np.dot is the matrix multiplication
    logprobs = logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    return cost


def update_parameters(parameters, grads, learning_rate=1.2):
    '''

    :param parameters:
    :param grads:
    :param learning_rate:

    :return:
        parameters : the new parameters updated

    '''

    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]

    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    parameters = {
        "W2": W2,
        "b2": b2,
        "W1": W1,
        "b1": b1
    }

    return parameters


def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    '''

    :param X: the input data (shape dim, num)
    :param Y: the output, (shape 1, num)
    :param n_h: 4
    :param num_iterations:
    :param print_cost:

    :return:
        parameters: the final trained parameters
    '''

    np.random.seed(3)
    n_x, n_y = layer_sizes(X, Y)[0], layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.5)

        if print_cost:
            if i % 1000 == 0:
                print(f"iterstions num: {i} , cost: {cost}")

    return parameters

def predict(parameters, X):
    '''

    :param parameters:
    :param X:

    :return:
    '''

    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2) 

    return predictions