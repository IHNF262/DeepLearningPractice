import numpy as np
import rnn_utils
import cllm_utils

def rnn_cell_forward(xt, a_prev, parameters):
    '''
    根据图2实现RNN单元的单步前向传播

    参数：
        xt -- 时间步“t”输入的数据，维度为（n_x, m）
        a_prev -- 时间步“t - 1”的隐藏隐藏状态，维度为（n_a, m）
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a_next -- 下一个隐藏状态，维度为（n_a， m）
        yt_pred -- 在时间步“t”的预测，维度为（n_y， m）
        cache -- 反向传播需要的元组，包含了(a_next, a_prev, xt, parameters)

    '''

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    根据图3来实现循环神经网络的前向传播

    参数：
        x -- 输入的全部数据，维度为(n_x, m, T_x) , T_x是时间步的长度(即rnn神经单元的数量)
        a0 -- 初始化隐藏状态，维度为 (n_a, m)
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x), hidden state for every time-step
        y_pred -- 所有时间步的预测，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize the value
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)  # now, size(x) == T_x

        a[:, :, t] = a_next
        y_pred[:, :, t] = yt_pred

        caches.append(cache)

    # stord values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


def rnn_cell_backward(da_next, cache):
    '''
    实现基本的RNN单元的单步反向传播

    参数：
        da_next -- 关于下一个隐藏状态的损失的梯度。
        cache -- 字典类型，rnn_step_forward()的输出

    返回：
        gradients -- 字典，包含了以下参数：
                        dx -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 上一隐藏层的隐藏状态，维度为(n_a, m)
                        dWax -- 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
                        dba -- 偏置向量的梯度，维度为(n_a, 1)
    '''
    a_next, a_prev, xt, parameters = cache

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]

    ba = parameters["ba"]
    by = parameters["by"]

    # compute gradient of tanh
    dtanh = (1 - np.square(a_next)) * da_next

    dWax = np.dot(dtanh, xt.T)
    dWaa = np.dot(dtanh, a_prev.T)
    dxt = np.dot(Wax.T, dtanh)
    da_prev = np.dot(Waa.T, dtanh)
    dba = np.sum(dtanh, axis=1, keepdims=True)

    gradients = {
        "dWax": dWax,
        "dWaa": dWaa,
        "dxt": dxt,
        "da_prev": da_prev,
        "dba": dba
    }

    return gradients


def rnn_backward(da, caches):
    '''

    在整个输入数据序列上实现RNN的反向传播

    参数：
        da -- 所有隐藏状态的梯度，维度为(n_a, m, T_x) #个人猜想，这个应该是每个unit 从输出y_hat那里继承来的 对a的偏导（梯度）
        caches -- 包含向前传播的信息的元组

    返回：
        gradients -- 包含了梯度的字典：
                        dx -- 关于输入数据的梯度，维度为(n_x, m, T_x)
                        da0 -- 关于初始化隐藏状态的梯度，维度为(n_a, m)
                        dWax -- 关于输入权重的梯度，维度为(n_a, n_x)
                        dWaa -- 关于隐藏状态的权值的梯度，维度为(n_a, n_a)
                        dba -- 关于偏置的梯度，维度为(n_a, 1)
    '''
    caches, x = caches
    a1, a0, x1, parameters = caches[0]

    # 获取da与x1的维度信息
    n_x, m = x1.shape
    n_a, m, T_x = da.shape

    da_prevt = np.zeros([n_a, m])
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])

        dxt = gradients["dxt"]
        da_prevt = gradients["da_prev"]
        dWaxt = gradients["dWax"]
        dWaat = gradients["dWaa"]
        dbat = gradients["dba"]

        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # 将 da0设置为a的梯度，该梯度已通过所有时间步骤进行反向传播
    da0 = da_prevt

    gradients = {
        "dx": dx,
        "da0": da0,
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba
    }

    return gradients


def clip(gradients, maxValue):
    """
    使用maxValue来修剪梯度

    参数：
        gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内

    返回：
        gradients -- 修剪后的梯度
    """
    # 获取参数
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']

    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def sample(parameters, char_to_ix, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样

    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典
        seed -- 随机种子

    返回：
        indices -- 包含采样字符索引的长度为n的列表。
    """

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    b = parameters["b"] #ba
    by = parameters["by"]

    n_y = by.shape[0]
    n_a = Waa.shape[0]

    # initialize the value
    x = np.zeros((n_y, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    # （我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
    idx = -1
    counter = 0
    newLine_character = char_to_ix["\n"]

    while (idx != newLine_character and counter < 50):
        a = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        y = rnn_utils.softmax(np.dot(Wya, a) + by)

        np.random.seed(counter + seed)

        idx = np.random.choice(list(range(n_y)), p=y.ravel())  # RNN type: zero To many

        indices.append(idx)

        # generate new X
        x = np.zeros((n_y, 1))
        x[idx] = 1

        # update
        a_prev = a

        seed += 1
        counter += 1

    if counter == 50:
        indices.append(char_to_ix["\n"])

    return indices


def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
    """
    执行训练模型的单步优化。

    参数：
        X -- 整数列表，其中每个整数映射到词汇表中的字符。
        Y -- 整数列表，与X完全相同，但向左移动了一个索引。
        a_prev -- 上一个隐藏状态
        parameters -- 字典，包含了以下参数：
                        Wax -- 权重矩阵乘以输入，维度为(n_a, n_x)
                        Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                        Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                        b -- 偏置，维度为(n_a, 1)
                        by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
        learning_rate -- 模型学习的速率

    返回：
        loss -- 损失函数的值（交叉熵损失）
        gradients -- 字典，包含了以下参数：
                        dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                        dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                        db -- 偏置的梯度，维度为(n_a, 1)
                        dby -- 输出偏置向量的梯度，维度为(n_y, 1)
        a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
    """

    # 前向传播
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

    # 反向传播
    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients, 5)

    # 更新参数
    parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]

def model(data, ix_to_char, char_to_ix, num_iterations = 3500, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    训练模型并生成恐龙名字

    参数：
        data -- 语料库
        ix_to_char -- 索引映射字符字典
        char_to_ix -- 字符映射索引字典
        num_iterations -- 迭代次数
        n_a -- RNN单元数量
        dino_names -- 每次迭代中采样的数量
        vocab_size -- 在文本中的唯一字符的数量

    返回：
        parameters -- 学习后了的参数
    """
    n_x, n_y = vocab_size, vocab_size

    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)

    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

    # 构建恐龙名称列表
    with open("../dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)

    #loop
    a_prev = np.zeros((n_a, 1))
    for j in range(num_iterations):
        index = j % len(examples) #一次迭代只训练一个name
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = cllm_utils.smooth(loss, curr_loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 2000 == 0:
            print("number of iterations: " + str(j + 1) + "   ,  cost : " + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                cllm_utils.print_sample(sampled_indices, ix_to_char)

                # 为了得到相同的效果，随机种子+1
                seed += 1

            print("\n")
    return parameters








