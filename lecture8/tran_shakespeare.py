import time

import numpy as np

from lecture8.core import Dataset, DataLoader, accuracy, SGD
from lecture8.time_rnn import TimeSimpleRNN, time_softmax_cross_entropy


class TinyShakespeareDataset(Dataset):
    def __init__(self, seq_len=50, is_train=True, y_transform=None, t_transform=None):
        self.seq_length = seq_len
        self.is_train = is_train
        self.char_to_id = None
        self.id_to_char = None
        super().__init__(train=is_train, y_transform=y_transform, t_transform=t_transform)

    def prepare(self):
        file_path, origin_text = "shakespeare.txt", ""
        with open(file_path, "r") as f:
            origin_text = f.read()

        # 字符级编码
        chars = sorted(list(set(origin_text)))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

        encoded = np.array([self.char_to_id[c] for c in origin_text], dtype=np.int32)
        n = int(len(encoded) * 0.9)  # 测试集和训练集的划分点
        if self.is_train:
            self.data = encoded[:n]
        else:
            self.data = encoded[n:]
        # 截断为 seq_len 的整数倍
        total_len = len(self.data) // self.seq_length * self.seq_length
        self.data = self.data[:total_len]

    # idx 指的是从 idx 开始的 时序数据块[块大小为 seq_length]
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        chunk = self.data[start:end]
        x = chunk[:-1]
        t = chunk[1:]
        return x, t

    def __len__(self):
        return len(self.data) // self.seq_length


if __name__ == "__main__":
    # 设置一些超参数
    hidden_size = 16
    seq_length = 50
    batch_size = 128
    epochs = 5
    learning_rate = 0.1

    train_set = TinyShakespeareDataset(seq_len=seq_length, is_train=True)
    test_set = TinyShakespeareDataset(seq_len=seq_length, is_train=False)

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    output_size = len(train_set.char_to_id)  # 一般是 vocab_size

    #   输出采用数字编码，在计算损失时会将其转换为 one-hot 编码
    model = TimeSimpleRNN(hidden_size, output_size)
    optimizer = SGD(model, learning_rate)

    # 检查一批样本
    for x, t in train_loader:
        print("一个rnn块的输入序列（整数编码）:", x.shape)
        print("一个rnn块的目标序列（整数编码）:", t.shape)
        break

    for epoch in range(epochs):
        # ---- 训练阶段 ----
        model.reset_state()
        total_loss, total_acc = 0, 0
        count = 0
        for x, t in train_loader:
            s1 = int(round(time.time() * 1000))
            y = model(x)
            s2 = int(round(time.time() * 1000))
            print("正向传播耗时[毫秒]: ", s2 - s1)
            loss = time_softmax_cross_entropy(y, t)
            s3 = int(round(time.time() * 1000))
            print("带loss正向传播耗时[毫秒]: ", s3 - s2)
            model.clear_grads()
            loss.backward()
            s4 = int(round(time.time() * 1000))
            print("反向传播耗时[毫秒]: ", s4 - s3)
            optimizer.update()
            s5 = int(round(time.time() * 1000))
            print("参数更新耗时[毫秒]: ", s5 - s4)
            total_loss += loss.value * len(t)
            total_acc += accuracy(y, t).value * len(t)
            count += 1
            print(count)

        avg_train_loss = total_loss.value / len(train_loader.dataset)
        avg_train_acc = total_acc.value / len(train_loader.dataset)
        print("epoch", epoch, avg_train_loss, avg_train_acc)

        # ---- 测试阶段 ----
        total_loss, total_acc = 0, 0
        for x, t in test_loader:
            y = model(x)
            loss = time_softmax_cross_entropy(y, t)
