import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils #参见数据包或者在本文底部copy
import testCase  #参见数据包或者在本文底部copy

from model import *

train_X, train_Y = opt_utils.load_dataset(is_plot=True)
plt.show()


layers_dims = [train_X.shape[0], 5, 2, 1]

print("----------------- Mini-batch Gradient Descent -------------------------")
## model with gradient descent optimization
# parameters = model(train_X, train_Y, layers_dims, optimizer="gd")
# title = "Model with Gradient Descent optimization"

## model with momentum optimization
# parameters = model(train_X, train_Y, layers_dims, optimizer="momentum", beta=0.9)
# title = "Model with momentum optimization"

## model with adam optimization
parameters = model(train_X, train_Y, layers_dims, optimizer="adam")
title = "Model with Adam optimization"



# Predict
predictions = opt_utils.predict(train_X, train_Y,parameters)

# Plot decision boundary
plt.title(title)
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

plt.show()


