import matplotlib.pyplot as plt
import Other as tools
import numpy as np
from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

# plt.subplot(2,3,1)
# plt.imshow(train_set_x_orig[25])
# plt.subplot(2,3,2)
# plt.imshow(train_set_x_orig[1])
# plt.subplot(2,3,3)
# plt.imshow(train_set_x_orig[2])
# plt.subplot(2,3,4)
# plt.imshow(train_set_x_orig[3])
# plt.subplot(2,3,5)
# plt.imshow(train_set_x_orig[4])
# plt.subplot(2,3,6)
# plt.imshow(train_set_x_orig[5])
#plt.show()

m_train = train_set_y_orig.shape[1] #训练集里图片的数量。
m_test = test_set_y_orig.shape[1] #测试集里图片的数量。
num_px = train_set_x_orig.shape[1] #训练、测试集里面的图片的宽度和高度（均为64x64）。

print ("The number of training set: m_train = " + str(m_train))
print ("The number of test set : m_test = " + str(m_test))
print ("The width/height of each image : num_px = " + str(num_px))
print ("The shape of each image : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("The dimensions of training set _ image : " + str(train_set_x_orig.shape))
print ("The dimensions of training set _ label : " + str(train_set_y_orig.shape))
print ("The dimensions of test set _ image : " + str(test_set_x_orig.shape))
print ("The dimensions of test set _ label : " + str(test_set_y_orig.shape))

train_set_x_flatten =  train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("The dimensions of the training set after flatten _ image : " + str(train_set_x_flatten.shape))
print ("The dimensions of the training set after flatten _ label : " + str(train_set_y_orig.shape))
print ("The dimensions of the test set after flatten _ image : " + str(test_set_x_flatten.shape))
print ("The dimensions of the test set after flatten _ label : " + str(test_set_y_orig.shape))

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

computeMultipleLearningRates = False

if computeMultipleLearningRates == False:
    print("====================test model on a specific learning rate====================")
    # 这里加载的是真实的数据，请参见上面的代码部分。
    d = tools.model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

    costs = np.squeeze(d["costs"])

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title(f"Learning rate = {d['learning_rate']}")
    plt.show()
else:
    print("====================test model on different learning rates ====================")
    learning_rates = [0.005, 0.01, 0.001, 0.0001]
    models = {}

    for i in learning_rates:
        print(f"learning_rate is {i}")
        print("num iterations: 2000")
        models[i] = tools.model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = i, print_cost = True)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[i]['costs']), label=models[i]['learning_rate'])

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc = 'upper center', shadow = True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    plt.show()