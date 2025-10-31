import math

import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset  # 原始数据集
        self.iteration = 0  # 当前迭代次数
        self.index = None  # 当前批次的样本索引
        self.batch_size = batch_size  # 每个批次的样本数量
        self.shuffle = shuffle  # 是否在每个 epoch 开始时打乱数据索引
        self.data_size = len(dataset)
        # 每个 epoch 中，迭代的最大次数
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    # 重置迭代器，将迭代次数设为0，根据shuffle参数是否为True，重新设置样本索引
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size: (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
