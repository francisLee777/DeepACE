import numpy as np

from lecture6.core import as_variable, exp, sum, Function, Variable, log


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = x[self.slices]
        return y

    def backward(self, dy):
        # 构造一个与原始输入相同形状的 0 数组
        dx = np.zeros(self.x_shape, dtype=dy.dtype)
        # np.add.at 可以实现“稀疏加法”（用于切片梯度还原）
        np.add.at(dx, self.slices, dy.value)
        return Variable(dx)


def get_item(x, slices):
    return GetItem(slices)(x)


class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None
        self.argmax = None

    def forward(self, x):
        # 使用 np 的 max 函数计算最大值，注意指定轴方向和是否保持维度
        y = np.max(x, axis=self.axis, keepdims=self.keepdims)
        self.x_shape = x.shape  # 记录最大值的位置，用于反向传播
        if self.axis is None:
            # (x == y) 得到的是一个布尔数组，形状与 x 相同
            # 它的每个元素都是 True 或 False，对应 x 中是否等于最大值 y
            self.argmax = x == y
        else:
            self.argmax = x == np.expand_dims(y, axis=self.axis)
        return y

    def backward(self, dy):
        # dy 是上游梯度
        # 传播到所有最大值位置（可能有多个最大值相等）
        dx = dy * self.argmax.astype(dy.dtype)
        if not self.keepdims and self.axis is not None:
            # 当 keepdims=False 时，dy 的维度比 x 小
            dy = np.expand_dims(dy, axis=self.axis)
            dx = dy * self.argmax.astype(dy.dtype)
        return dx


def max(x, axis=None, keepdims=False):
    x = as_variable(x)
    return Max(axis, keepdims)(x)


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        return np.clip(x, self.x_min, self.x_max)

    def backward(self, dy):
        (x,) = self.input_variable
        mask = (x.value >= self.x_min) * (x.value <= self.x_max)
        dx = dy * mask
        return dx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]  # 一般 x 的第一个维度是批量数据个数 batch size
    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)  # 防止0和1溢出问题
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.value]
    return -1 * sum(tlog_p) / N


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        # 防止数据溢出，进行缩放
        # y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(x)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = y * dy
        sum_dx = dx.sum(axis=self.axis, keepdims=True)
        dx -= y * sum_dx
        return dx


class SoftmaxCrossEntropy(Function):
    def __init__(self):
        self.y = None  # 预测值
        self.t = None  # 标签值
        self.N = None  # 数据 batch 的大小

    def forward(self, x, t):
        # --- 数值稳定处理 ---
        # x = x - x.max(axis=1, keepdims=True)
        # --- softmax ---
        exp_x = np.exp(x)
        sum_exp_x = exp_x.sum(axis=1, keepdims=True)
        self.y = exp_x / sum_exp_x  # softmax输出

        self.t = t  # 保存标签
        self.N = x.shape[0]  # batch大小

        # --- clip避免 log(0) 和 log(1) ---
        log_p = np.log(np.clip(self.y, 1e-15, 1.0))

        # --- 交叉熵 ---
        loss = -np.sum(log_p[np.arange(self.N), t.ravel()]) / self.N
        return np.array(loss)

    def backward(self, dy):
        # 反向传播： dL/dx = (y - one_hot(t)) / N 注意：不要单独反传 softmax！
        # 拿到保存的值
        y, t, N = self.y, self.t, self.N
        # 构造 one-hot
        one_hot = np.zeros_like(y)
        one_hot[np.arange(N), t] = 1
        # softmax + crossentropy 的合成梯度
        dx = (y - one_hot) / N
        # 这个Node是指 t 无需梯度，因为t是标签，不是中间变量。如果没有None的话导致框架后续逻辑不兼容
        return dy * dx, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def softmax(x, axis=1):
    return Softmax(axis)(x)


if __name__ == "__main__":
    a = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = a[1]
    y.backward()
    print(y, a.grad)  # variable([4 5 6]) variable([[0 0 0] [1 1 1]])

    loss = softmax_cross_entropy_simple(np.random.rand(4, 10), np.array([2, 6, 9, 1]))
    print(loss)

    print("----------------------")
    a = Variable(np.array([[1, 1, 2], [3, 3, 3]]))
    y = softmax(a)
    y.backward()
    print(y, a.grad)

    print("----------------------")
    a = Variable(np.array([[1, 1, 2], [3, 3, 3]]))
    y = softmax_cross_entropy(a, np.array([2, 1]))
    y.backward()
    print(y, a.grad)
