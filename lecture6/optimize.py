import numpy as np

from lecture6.TwoLayerNet import TwoLayerNet
from lecture6.core import mean_squared_error


class Optimizer:
    def __init__(self, model, lr=0.01):
        self.lr = lr
        self.target = model  # 也可以传入 Layer 类
        self.hooks = []  # 钩子函数[可选]

    def add_hook(self, hook):
        self.hooks.append(hook)

    def update(self):
        params = self.target.params()
        # 过滤掉梯度为 None 的参数
        params = [p for p in params if p.grad is not None]

        # 调用钩子函数[可选]，可用于权重衰减、梯度裁剪等工作
        for hook in self.hooks:
            hook(params)

        # 逐个更新参数
        for param in params:
            self.update_one(param)

    # 每个参数的更新方法，需要在子类中实现
    def update_one(self, param):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model)
        self.lr = lr

    def update_one(self, param):
        param.value -= self.lr * param.grad.value


class AdaGrad(Optimizer):
    def __init__(self, model, lr=0.01, eps=1e-8):
        super().__init__(model)
        self.lr = lr
        self.eps = eps
        self.h = {}  # 累积平方梯度

    def update_one(self, param):
        if param.grad is None:
            return

        grad = param.grad.value

        # 初始化累积项
        if param not in self.h:
            self.h[param] = np.zeros_like(grad)

        # 累积梯度平方
        h = self.h[param]
        h += grad * grad

        # 更新参数
        param.value -= self.lr * grad / (np.sqrt(h) + self.eps)


class Momentum(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        self.v = {}  # 保存每个参数的动量项

    def update_one(self, param):
        if param.grad is None:
            return

        grad = param.grad.value

        # 初始化动量
        if param not in self.v:
            self.v[param] = np.zeros_like(grad)

        v = self.v[param]

        # 计算动量更新
        v[:] = self.momentum * v - self.lr * grad

        # 参数更新
        param.value += v


if __name__ == "__main__":
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

    # 设置一些超参数
    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)
    optimizer = Momentum(model, lr)

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
        optimizer.update()

        if i % 100 == 0:
            print(f"迭代 {i}: 损失 {loss.value:.4f}")
