import math

import numpy as np
from matplotlib import pyplot as plt

from lecture7.core import MLP, SGD, softmax_cross_entropy
from lecture7.core import as_array, Variable, as_variable
from lecture7.data_loader import DataLoader
from lecture7.dataset import ThreeClassDataset


def accuracy(y, t):
    # 输入需要是 one-hot 编码
    y, t = as_variable(y), as_variable(t)
    # 预测值中概率最大的类别，构成与标签相同的形状
    pred = y.value.argmax(axis=1).reshape(t.shape)
    result = pred == t.value  # 预测值与标签值相等的位置为True，否则为False
    acc = result.mean()  # 计算准确率，即正确预测的样本数占总样本数的比例
    return Variable(as_array(acc))


if __name__ == "__main__":
    y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
    t = np.array([1, 2, 0])
    acc = accuracy(y, t)
    print(acc)  # variable(0.6666666666666666)

    max_epoch = 300  # 最大训练轮数
    batch_size = 30  # 每个批次的样本数量
    hidden_size = 10  # 隐藏层神经元数量
    lr = 1.0  # 学习率

    train_set = ThreeClassDataset()
    # 定义模型，输入层到隐藏层有hidden_size个神经元，隐藏层到输出层有3个神经元
    model = MLP((hidden_size, 3))
    optimizer = SGD(model, lr)  # SGD 优化器

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)  # 每个 epoch 中，迭代的最大次数
    #
    for epoch in range(max_epoch):
        # 打乱数据索引，使每次训练的样本顺序不同
        index = np.random.permutation(data_size)
        sum_loss = 0

        # 每次迭代取出一个批次的样本，进行前向传播、计算损失、反向传播、更新参数
        for i in range(max_iter):
            batch_index = index[i * batch_size: (i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            y = model(batch_x)
            loss = softmax_cross_entropy(y, batch_t)
            model.clear_grads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.value) * len(batch_t)

            # acc = accuracy(y, batch_t)
            # print("iter %d, accuracy %.2f" % (epoch + 1, float(acc.value)))

        # Print loss every epoch
        avg_loss = sum_loss / data_size
        print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))

    # 使用 DataLoader 进行训练
    train_set = ThreeClassDataset(is_train=True)
    test_set = ThreeClassDataset(is_train=False)

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 3))
    optimizer = SGD(model, lr)

    # for 数据可视化
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []

    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0
        # 训练
        for x, t in train_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            model.clear_grads()
            loss.backward()
            optimizer.update()
            acc = accuracy(y, t)
            sum_loss += float(loss.value) * len(t)
            sum_acc += float(acc.value) * len(t)

        print("epoch: {}".format(epoch + 1))
        print("train loss: {:.4f}, accuracy: {:.4f}".format(sum_loss / len(train_set), sum_acc / len(train_set)))

        # 每一轮训练结束后，使用测试集评估模型性能
        sum_loss_test, sum_acc_test = 0, 0
        for x, t in test_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            sum_loss_test += float(loss.value) * len(t)
            sum_acc_test += float(acc.value) * len(t)

        print("test loss: {:.4f}, accuracy: {:.4f}".format(sum_loss_test / len(test_set), sum_acc_test / len(test_set)))

        # ======= 保存数据用于可视化 =======
        train_loss_list.append(sum_loss / len(train_set))
        test_loss_list.append(sum_loss_test / len(test_set))
        train_acc_list.append(sum_acc / len(train_set))
        test_acc_list.append(sum_acc_test / len(test_set))

    # ======= 绘图 =======
    epochs = range(1, max_epoch + 1)
    plt.figure(figsize=(12, 5))

    # --- loss 曲线 ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, test_loss_list, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)

    # --- acc 曲线 ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, test_acc_list, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
