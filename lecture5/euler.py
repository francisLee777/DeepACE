import numpy as np

from lecture5.core import matmul, sum, Variable, exp, Function

# 训练数据，从 -3 到 3 等间隔取 100 个点，然后 reshape 成 100 * 1 的向量
x = Variable(np.linspace(0, 3, 100).reshape(100, 1))
y = exp(x)  # 真实值

# 简单的两层网络
W1 = Variable(0.01 * np.random.randn(1, 100))
b1 = Variable(np.zeros(100))
W2 = Variable(0.01 * np.random.randn(100, 1))
b2 = Variable(np.zeros(1))


class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = dy * y * (1 - y)
        return dx


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    y = 1 / (1 + exp(-x))
    return y


# def relu(z):
#     return np.maximum(0, z)


def mean_squared_error(x0, x1):
    # temp = np.max(np.abs(x0.value))
    # x0 = x0 / temp  # 等比缩放，不然会导致 sum(diff ** 2) 溢出
    # temp2 = np.max(np.abs(x1.value))
    # x1 = x1 / temp2  # 等比缩放，不然会导致 sum(diff ** 2) 溢出
    diff = x1 - x0
    return sum(diff**2) / len(
        diff
    )  # 计算均方误差，除以样本数量, 防止误差过大溢出以及学习率无法调整


def abs_loss(x0, x1):
    diff = abs(x1 - x0)
    return sum(diff) / len(diff)  # 除以样本数量, 防止误差过大溢出以及学习率无法调整


def predict(x):
    temp = matmul(x, W1) + b1
    temp = sigmoid(temp)
    result = matmul(temp, W2) + b2
    return result


lr = 0.2  # 学习率
iters = 10000  # 迭代次数

for epoch in range(iters):
    y_predit = predict(x)
    loss = abs_loss(y, y_predit)

    loss.backward()  # 损失函数反向传播

    W1.value -= lr * W1.grad.value
    b1.value -= lr * b1.grad.value
    W2.value -= lr * W2.grad.value
    b2.value -= lr * b2.grad.value

    W1.grad = None  # 每次迭代后，需要将梯度重置为 0，否则会影响下一次迭代
    b1.grad = None
    W2.grad = None  # 每次迭代后，需要将梯度重置为 0，否则会影响下一次迭代
    b2.grad = None

    if epoch % 100 == 0:  # 每100次，打印输出一下损失值
        print(f"{epoch}: loss={loss.value:.4f}")

test_x = 5
print(np.exp(test_x))
print(predict(Variable(np.array([test_x]))))
