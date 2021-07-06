import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

import reg_utils  # 第二部分，正则化
import gc_utils  # 第三部分，梯度校验

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=False)

# initialize parameters without regularization
# parameters = reg_utils.model(train_X, train_Y, is_plot=True)
# initTitle = "Model without regularization"

# initialize parameters with L2-regularization
# parameters = reg_utils.model(train_X, train_Y, lambd=0.7, is_plot=True)
# initTitle = "Model with L2-regularization"

parameters = reg_utils.model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=True)
initTitle = "Model with Dropout-regularization"

print(f"----------------------  {initTitle} -------------------------")

print("\ntraining sets:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("test sets:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

plt.title(initTitle)
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# A2 = np.array([[1, 2, 3],
#               [2, 4, -1]])
#
# print(f"A2 shape {A2.shape}")
# print(f"A2 :{A2}")
# print(f"int64+ {np.int64(A2 > 0)}")