import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    '''
    initilize the w and b
    :param dim:
    :return:
    '''
    w = np.zeros(shape=(dim, 1))
    b = 0

    return (w, b)


def propagate(w, b, X, Y):
    '''
    w : 12000,1
    b : 1,1
    X : 12000, 209
    Y: 1, 209
    '''

    # print(w.shape)
    # print(X.shape)
    # print(Y.shape)

    m = X.shape[1]  # m=209

    A = sigmoid(np.dot(w.T, X) + b)  # shape 1, 209
    cost = (-1 / m) * np.sum((Y * np.log(A) + (1 - Y) * (np.log(1 - A))))  # shape 1,1

    dw = (1 / m) * np.dot(X, (A - Y).T)  # shape 12000,1
    db = (1 / m) * np.sum(A - Y)  # shape 1,1

    cost = np.squeeze(cost)  # shape 1， erase extra one dimension data of array

    grads = {
        "dw": dw,
        "db": db
    }

    return (grads, cost)


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=True):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if (print_cost) and i % 100 == 0:
            print(f"iteration number {i} , cost: {cost}")

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return (params, grads, costs)


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 100, learning_rate = 0.009,  print_cost = True):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w, b = parameters["w"], parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print(f"Accuracy of training set: {100 - np.mean(np.abs(Y_prediction_train - Y_train))*100} %")
    print(f"Accuracy of test set: {100 - np.mean(np.abs(Y_prediction_test - Y_test))*100} %")

    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d
