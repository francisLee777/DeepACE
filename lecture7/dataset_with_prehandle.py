# 带预处理的数据集
import numpy as np


class Dataset:
    def __init__(self, train=True, y_transform=None, t_transform=None):
        self.train = train
        self.y_transform = y_transform  # 样本预处理函数，可为 None
        self.t_transform = t_transform  # 标签预处理函数，可为 None

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        y = self.data[index]
        t = None if self.label is None else self.label[index]
        # 延迟 transform 判断（只有在存在时才调用）
        if self.y_transform:
            y = self.y_transform(y)
        if self.t_transform and t is not None:
            t = self.t_transform(t)
        return y, t

    def __len__(self):
        return len(self.data)

    def prepare(self):
        """由子类实现：生成或加载数据"""
        pass


# 测试预处理函数：将样本/标签值除以2
def transform_div2(x):
    return x / 2


if __name__ == "__main__":
    # 测试数据集
    class TestDataset(Dataset):
        def prepare(self):
            self.data = np.random.randn(100, 2)
            self.label = np.random.randint(0, 2, size=100)

    # 测试预处理函数
    def y_transform(x):
        return x * 2

    def t_transform(t):
        return t + 1

    # 创建测试数据集
    test_dataset = TestDataset(
        train=True, y_transform=transform_div2, t_transform=transform_div2
    )

    # 测试 __getitem__
    y, t = test_dataset[0]
    print("y:", y)
    print("t:", t)

    # 测试 __len__
    print("Dataset length:", len(test_dataset))
