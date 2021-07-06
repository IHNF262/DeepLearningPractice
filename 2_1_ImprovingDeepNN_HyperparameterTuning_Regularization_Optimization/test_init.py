
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   #第一部分，初始化


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=False)

# # initialize the parameters with zeros
# parameters = init_utils.initialize_parameters_zeros([3,2,1])
# initType = "zeros"

# # initialize the parameters with random
# parameters = init_utils.initialize_parameters_random([3,2,1])
# initType = "random"

# initialize the parameters with he(multiplied by one sqrt())
parameters = init_utils.initialize_parameters_he([3,2,1])
initType = "he"

print(f"------------------------- initialize with {initType} ------------------------------")

print("----------------------- Test initialize_parameters-------------------------")
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

print("----------------------- Test Model  -------------------------")
parameters = init_utils.model(train_X, train_Y, initialization=initType, is_plot=True)

print("\ntraining sets: ")
prediction_train = init_utils.predict(train_X, train_Y, parameters)
print("test sets: ")
prediction_test = init_utils.predict(test_X, test_Y, parameters)

print(f"\npredictions_train = {prediction_train}")
print(f"predictions_test = {prediction_test}")

plt.title(f"Model with {initType} initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
