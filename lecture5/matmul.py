import numpy as np

from lecture5.core import Function, Variable


class MatMul(Function):
    def forward(self, input_x, input_W):
        return input_x.dot(input_W)

    def backward(self, dy):
        # 入参是 ndarray
        input_x, input_w = self.input_variable
        # 1. 要使用 matmul 而不是 dot , 不然会有拆包和组装 Variable 类型的问题
        # 2. 这里 .T 的写法是转置操作，在之前的 Variable 类中 transpose 函数中已经实现过了
        dx = matmul(dy, input_w.T)
        dW = matmul(input_x.T, dy)
        # 出参是 Variable 类型
        return dx, dW


def matmul(input_x, input_W):
    return MatMul()(input_x, input_W)


if __name__ == '__main__':
    # 向量的内积
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.dot(a, b)
    print(c)  # 32
    # 矩阵的乘积
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = np.dot(a, b)
    print(c)  # [[19 22] [43 50]]
    d = np.dot(b, a)  # b 点乘 a 得到的结果不一样
    print(d)  # [[23 34] [31 46]]

    x = Variable(np.array([[1, 2]]))
    W = Variable(np.array([[5, 6], [7, 8]]))
    y = matmul(x, W)
    y.backward()
    print(x.shape, W.shape)
    print(y)  # variable([[19 22]])
    print(x.grad)  # variable([[11 15]])
    print(W.grad)  # variable([[1 1][2 2]])
