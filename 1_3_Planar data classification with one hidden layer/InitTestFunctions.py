import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from model import *

# plotRow,plotCol = 2,2

# X, Y = load_planar_dataset()  # initialize a scattered data set ,X is the 2D data,  Y is the Label(0 or 1)
# plt.subplot(plotRow, plotCol, 1)
# plt.scatter(X[0, :], X[1, :], c=Y, s=20, cmap=plt.cm.Spectral)  # show the scatter, s is the size of a point in scatter
#
# plt.show()

#test layer_sizes
print("=========================test layer_sizes=========================")
X_asses , Y_asses = layer_sizes_test_case()
(n_x,n_h,n_y) =  layer_sizes(X_asses,Y_asses)
print("the number of nodes in this input layer: n_x = " + str(n_x))
print("the number of nodes in this hidden layer: n_h = " + str(n_h))
print("the number of nodes in this output layer: n_y = " + str(n_y))

#test initialize_parameters
print("=========================test initialize_parameters=========================")
n_x , n_h , n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x , n_h , n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#test forward_propagation
print("=========================test forward_propagation=========================")
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)
print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))

#test compute_cost
print("=========================test compute_cost=========================")
A2 , Y_assess , parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2,Y_assess,parameters)))

#test backward_propagation
print("=========================test backward_propagation=========================")
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

#test update_parameters
print("=========================test update_parameters=========================")
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))



# #test nn_model
# print("=========================test nn_model=========================")
# X_assess, Y_assess = nn_model_test_case()
#
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
#
#test predict
print("=========================test predict=========================")

parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("the mean value of prediction = " + str(np.mean(predictions)))
