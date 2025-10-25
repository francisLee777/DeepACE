import numpy as np

from lecture5.core import Variable, matmul, sum

if __name__ == '__main__':
    # 随机生成一个形状为 (100, 1) 的数组， 其中的每个元素都是大小 [0, 1) 浮点数
    x = np.random.rand(100, 1)
    y = 30 * x + 50 + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)  # 可以省略, 因为在 DeepACE 中的函数 __call__ 逻辑，都处理了将 np.ndarray 转化为 Variable 类型
    W = Variable(np.zeros((1, 1)))  # 初始化权重为 0
    b = Variable(np.zeros(1))  # 初始化偏置为 0


    def predict(x):
        return matmul(x, W) + b


    def mean_squared_error(x0, x1):
        diff = x1 - x0
        return sum(diff ** 2) / len(diff)  # 计算均方误差，除以样本数量, 防止误差过大溢出以及学习率无法调整


    lr = 0.1  # 学习率
    iters = 100  # 迭代次数

    for i in range(iters):
        y_predit = predict(x)
        loss = mean_squared_error(y, y_predit)

        loss.backward()  # 损失函数反向传播

        W.value -= lr * W.grad.value
        b.value -= lr * b.grad.value

        W.grad = None  # 每次迭代后，需要将梯度重置为 0，否则会影响下一次迭代
        b.grad = None

    print("W:", W.value)
    print("b:", b.value)
