import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from model import *

plotRow,plotCol = 1,2
plt.figure(figsize=(12,6))
X, Y = load_planar_dataset()  # initialize a scattered data set ,X is the 2D data,  Y is the Label(0 or 1)
plt.subplot(plotRow, plotCol, 1)
plt.scatter(X[0, :], X[1, :], c=Y, s=15, cmap=plt.cm.Spectral)  # show the scatter, s is the size of a point in scatter

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")


print(" -------------------------- logistic regression test --------------------------")
test0 = True
# test0 = False
if test0:
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    plt.subplot(plotRow, plotCol, 2)
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # draw boundary
    plt.title("Logistic Regression")
    LR_predictions = clf.predict(X.T)
    print(f"Logistical regression accuracy of the training set : {float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100)} % ")



print(" \n-------------------------- nn test --------------------------")
plt.figure()
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

#draw boundary
# plt.subplot(plotRow, plotCol, 3)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ('nn accruacy of the training set: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')


# print(" \n-------------------------- nn test on different hidden units --------------------------")
plotRow,plotCol = 4,2
plt.figure(figsize=(16, 20))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #the number of units in the hidden layer
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(plotRow, plotCol, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=10000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("the number of units in hidden layer ： {}  ，Accuracy: {} %".format(n_h, accuracy))

plt.show()