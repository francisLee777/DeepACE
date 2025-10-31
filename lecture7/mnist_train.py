# pip install mnist

import mnist
import numpy as np
from matplotlib import pyplot as plt

from lecture7.core import MLP, softmax_cross_entropy, SGD
from lecture7.data_loader import DataLoader
from lecture7.data_train import accuracy
from lecture7.dataset_with_prehandle import Dataset

# 下载文件的目录，取当前文件夹，否则默认情况下在系统的 tmp 目录
mnist.temporary_dir = lambda: './'
# 默认的数据源url好像失效了，使用下面的url
mnist.datasets_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'


class MNISTDataset(Dataset):
    def prepare(self):
        if self.train:
            images = mnist.train_images()  # shape: (60000, 28, 28)
            labels = mnist.train_labels()  # shape: (60000,)
        else:
            images = mnist.test_images()  # shape: (10000, 28, 28)
            labels = mnist.test_labels()  # shape: (10000,)

        # 数据预处理
        # 转为 float32 并归一化到 [0, 1]
        images = images.astype(np.float32) / 255.0

        # 拉平成 (N, 784)，方便输入 MLP
        images = images.reshape(len(images), -1)

        self.data = images
        self.label = labels


def train(model, optimizer, train_loader, test_loader, epochs):
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        total_loss, total_acc = 0, 0
        for x, t in train_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            model.clear_grads()
            loss.backward()
            optimizer.update()
            total_loss += loss * len(t)
            total_acc += accuracy(y, t) * len(t)

        avg_train_loss = total_loss.value / len(train_loader.dataset)
        avg_train_acc = total_acc.value / len(train_loader.dataset)

        # ---- 测试阶段 ----
        total_loss, total_acc = 0, 0
        for x, t in test_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            total_loss += loss * len(t)
            total_acc += accuracy(y, t) * len(t)

        avg_test_loss = total_loss.value / len(test_loader.dataset)
        avg_test_acc = total_acc.value / len(test_loader.dataset)

        print(f"Epoch {epoch + 1:3d}: "
              f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.4f}, "
              f"test_loss={avg_test_loss:.4f}, test_acc={avg_test_acc:.4f}")

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        test_loss_list.append(avg_test_loss)
        test_acc_list.append(avg_test_acc)

    return train_loss_list, test_loss_list, train_acc_list, test_acc_list


if __name__ == '__main__':
    input_size = 784
    hidden_size = 64
    output_size = 10  # 手写数字，10分类
    batch_size = 128
    lr = 0.1
    epochs = 10

    train_set = MNISTDataset(train=True)
    test_set = MNISTDataset(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MLP((hidden_size, output_size))
    optimizer = SGD(model, lr)
    train_loss, test_loss, train_acc, test_acc = train(model, optimizer, train_loader, test_loader, epochs)

    # ---- 可视化 ----
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(test_acc, label="Test Acc")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
