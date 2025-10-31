import numpy as np


class Dataset:
    def __init__(self, is_train=True):
        self.is_train = is_train  # 是训练 or 测试
        self.data = None
        self.label = None  # 标签值，可选。例如无监督学习不需要标签
        self.prepare()

    # 获取数据集的样本的方式
    def __getitem__(self, index):
        assert np.isscalar(index)  # index 必须是标量
        if self.label is None:
            return self.data[index], None
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    # 准备数据集，例如读取数据、预处理等。可选
    def prepare(self):
        pass


def get_example_data(is_train=True):
    # 测试数据集和训练数据集的随机种子不同，用来模拟。
    np.random.seed(seed=(1 if is_train else 2))

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)  # 标签，每个类别对应一个整数

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle, 随机打乱数据
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]  # 用于输入数据集，每个类别对应一个整数
    t = t[indices]  # 标签数据集，每个类别对应一个整数
    return x, t


class ThreeClassDataset(Dataset):
    def prepare(self):
        self.data, self.label = get_example_data(self.is_train)


if __name__ == "__main__":
    # # ---- 绘制 ----
    # x, t = get_example_data()
    # plt.figure(figsize=(6, 6))
    # # 不同类别使用不同颜色
    # for i in range(3):
    #     plt.scatter(x[t == i, 0], x[t == i, 1], label=f"Class {i}", s=20)
    # plt.legend()
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.show()
    data_set = ThreeClassDataset()
    batch_index = [0, 2, 4]  # 取出第0个～第2个数据
    batch = [data_set[i] for i in batch_index]
    print(batch)  # [(data_0, label_0), (data_2, label_2), (data_4, label_4)]
    x = np.array([example[0] for example in batch])  # 转换为 numpy 数组
    t = np.array([example[1] for example in batch])  # 转换为 numpy 数组
    print(x.shape)  # (3, 2)
    print(t.shape)  # (3,)  标签值，非one-hot
