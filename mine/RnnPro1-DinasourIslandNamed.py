import numpy as np
import random
import time
import cllm_utils
import RnnModel

print("------------------------ open training-set -----------------------------")
data = open("../dinos.txt", "r").read()

data = data.lower()

chars = list(set(data)) #convert data into a list of unordered and non-repetitve elements
data_size, vocab_size = len(data), len(chars)

print(chars)
print(f"There are total {data_size} characters, of which {vocab_size} are only characters ")

print("------------------------- construct two dictionary --------------------------")
char_to_ix = {
    ch:i for i, ch in enumerate(sorted(chars))
}

ix_to_char = {
    i:ch for i, ch in enumerate(sorted(chars))
}

print(char_to_ix)
print(ix_to_char)

print("------------------------- Test sample --------------------------")
np.random.seed(2)
n_y, n_a = vocab_size, 100
Wax, Waa, Wya = np.random.randn(n_a, n_y), np.random.randn(n_a, n_a), np.random.randn(n_y, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(n_y, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = RnnModel.sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])

print("------------------------- Test optimize --------------------------")
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = RnnModel.optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])

print("------------------------- Test Model --------------------------")

#开始时间
start_time = time.time()
#开始训练
parameters = RnnModel.model(data, ix_to_char, char_to_ix, num_iterations=30000)

#结束时间
end_time = time.time()

#计算时差
minium = end_time - start_time

print("compute time：" + str(int(minium / 60)) + " mins " + str(int(minium%60)) + " secs")
