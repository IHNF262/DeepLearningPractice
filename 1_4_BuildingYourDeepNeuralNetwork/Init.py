from lr_utils import load_dataset
import lr_utils
import model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

print("----------------- Test: Two layer NN ----------------------")

n_x = 12288
n_h = 7
n_y = 1

parameters = model.two_layer_model(train_x, train_set_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)

predictions_train = model.predict(train_x, train_y, parameters) #training sets
predictions_test = model.predict(test_x, test_y, parameters) #test sets

# print("----------------- Test: Four layer NN ----------------------")
# layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
# parameters = model.L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True,isPlot=True)
#
# predictions_train = model.predict(train_x, train_y, parameters) #training sets
# predictions_test = model.predict(test_x, test_y, parameters) #test sets
#
# model.print_mislabeled_images(classes, test_x, test_y, predictions_test)



# # print("----------------- Test: Recognize special Image ----------------------")
# my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
#
# num_px = 64
# image = Image.open('my_image.jpg')
#
# my_image = np.array(image.resize((num_px, num_px), Image.ANTIALIAS))
#
# my_image = my_image.reshape(num_px * num_px * 3, -1)
#
# predict_my_image = model.predict(my_image, my_label_y, parameters)
#
# plt.imshow(image)
#
# print("y = " + str(np.squeeze(predict_my_image)) + ", your L-layer model predicts a \"" + classes[
#     int(np.squeeze(predict_my_image))].decode("utf-8") + "\"picture.")
