import numpy as np

from lecture6.core import sigmoid, mean_squared_error
from lecture6.linear import Linear
from lecture6.model import Model


class TwoLayerNet(Model):
    def __init__(self, hidden_size, output_size, dtype=np.float32):
        super().__init__()
        self.l1 = Linear(hidden_size, dtype=dtype)
        self.l2 = Linear(output_size, dtype=dtype)

    def forward(self, x):
        h = sigmoid(self.l1(x))
        temp = self.l2(h)
        return temp


if __name__ == "__main__":
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

    # 设置一些超参数
    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)

    # 开始训练
    for i in range(max_iter):
        # 前向传播
        y_predit = model(x)
        # 计算损失
        loss = mean_squared_error(y_predit, y)

        # 重置权重并反向传播
        model.clear_grads()
        loss.backward()

        # 更新参数
        for param in model.params():
            param.value -= lr * param.grad.value

        if i % 100 == 0:
            print(f"迭代 {i}: 损失 {loss.value:.4f}")

    # 绘制计算图
    model.plot(x)
