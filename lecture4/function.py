import numpy as np

from lecture4.core import Variable, Function, as_variable, as_array


# 本节新增的函数。合并到 core.py 中

class Reshape(Function):
    def __init__(self, target_shape):
        self.origin_shape = None  # 先声明
        self.target_shape = target_shape

    def forward(self, x):
        self.origin_shape = x.shape
        return np.reshape(x, self.target_shape)

    def backward(self, dy):
        return np.reshape(dy, self.origin_shape)  # 反向传播时，需要将dy的形状恢复到x的形状


def reshape(input_x, shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return Reshape(shape)(as_array(input_x))


class Transpose(Function):
    def __init__(self, input_axes=None):
        self.axes = input_axes

    def forward(self, input_x):
        return np.transpose(input_x)

    def backward(self, dy):
        if self.axes is None:
            return transpose(dy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(dy, inv_axes)


def transpose(input_x, axes=None):
    return Transpose(axes)(as_array(input_x))


class Sum(Function):
    """
    沿指定轴计算张量的元素总和。
    """

    def __init__(self, axis=None, keep_dims=False):
        self.axis = axis
        self.keep_dims = keep_dims
        self.output_shape_kept = None
        self.origin_shape = None

    def forward(self, input_x):
        """
        执行前向传播。
        1. 保存输入形状 `self.origin_shape`，这对于反向传播至关重要。
        2. 计算并保存 `self.output_shape_kept`，即使 `keep_dims=False`，输出的形状。
        3. 使用 np.sum 执行实际的求和操作。
        """
        self.origin_shape = input_x.shape

        # 计算 "keep_dims=True" 时的输出形状
        if self.axis is None:
            self.output_shape_kept = (1,) * input_x.ndim
        else:
            # 处理 axis 为 int 或 tuple 的情况
            if isinstance(self.axis, int):
                axis_tuple = (self.axis,)
            else:
                axis_tuple = self.axis

            # 归一化轴索引（确保为正整数）
            normalized_axis = [ax % input_x.ndim for ax in axis_tuple]
            shape_list = list(input_x.shape)
            for ax in normalized_axis:
                shape_list[ax] = 1
            self.output_shape_kept = tuple(shape_list)

        # 执行求和操作
        y = np.sum(input_x, axis=self.axis, keepdims=self.keep_dims)
        return y

    def backward(self, gy):
        """
        执行反向传播。

        1. 通过 reshape 调整梯度形状。
        2. 使用广播机制将梯度广播回原始输入形状。
        """

        # 将梯度 reshape 为 "keep_dims=True" 时的形状
        gy_reshaped = np.reshape(gy, self.output_shape_kept)

        # 将梯度广播回原始形状
        gx = broadcast_to(gy_reshaped, self.origin_shape)
        return gx


def sum(input_x, axis=None, keep_dims=False):
    return Sum(axis, keep_dims)(input_x)


class BroadcastTo(Function):
    def __init__(self, target_shape):
        self.origin_shape = None  # 先声明
        self.target_shape = target_shape

    def forward(self, input_x):
        self.origin_shape = input_x.shape
        return np.broadcast_to(input_x, self.target_shape)

    def backward(self, dy):
        return sum_to(dy, self.origin_shape)


def broadcast_to(input_x, shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return BroadcastTo(shape)(as_array(input_x))


class SumTo(Function):
    def __init__(self, target_shape):
        self.origin_shape = None
        self.target_shape = target_shape

    def forward(self, input_x):
        self.origin_shape = input_x.shape  # 保存原始形状
        return util_sum_to(input_x, self.target_shape)

    def backward(self, dy):
        return broadcast_to(dy, self.origin_shape)


def sum_to(input_x, shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return SumTo(shape)(as_array(input_x))


def util_sum_to(input_x, target_shape):
    y = input_x
    # 处理广播对齐过程中新增的维度：input_x 比 target_shape 多出来的“前导维度”（leading dimensions）
    while y.ndim > len(target_shape):
        y = y.sum(axis=0)
    # 对 shape=1 的维度求和。被拉伸的维度：target_shape 中为 1，但在 input_x 中被拉伸为 N 的维度。
    for i, sx in enumerate(target_shape):
        if sx == 1:
            y = y.sum(axis=i, keepdims=True)
    return y


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x.reshape((6,)))
    y1 = np.reshape(x, (6,))
    y2 = np.reshape(x, (3, 2))
    print(y1)  # [1 2 3 4 5 6]
    print(y2)  # [[1 2] [3 4] [5 6]]

    print('----')
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = reshape(x, (6,))
    z = y ** 2
    z.backward()
    print('z', z, 'z.grad', z.grad)  # z variable([ 1  4  9 16 25 36]) z.grad [1 1 1 1 1 1]
    print('y', y, 'y.grad', y.grad)  # y variable([1 2 3 4 5 6]) y.grad [ 2  4  6  8 10 12]
    print('x', x, 'x.grad', x.grad)  # x variable([[1 2 3] [4 5 6]]) x.grad [[ 2  4  6] [ 8 10 12]]

    y = x.reshape((3, 2))
    z = x.reshape(3, 2)
    print(y, z)

    print('----')
    y = transpose(Variable(np.array([[1, 2, 3], [4, 5, 6]])))
    z = transpose(Variable(np.array([[[1, 2, 3], [4, 5, 6]]])))

    print(y)  # variable([[1 4] [2 5] [3 6]])
    print(z)  # variable([[[1] [4]] [[2] [5]]  [[3] [6]]])

    print('----')
    y = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    z = Variable(np.array([[[1, 2, 3], [4, 5, 6]]]))
    print(y.transpose())  # variable([[1 4] [2 5] [3 6]])
    print(z.T)  # variable([[[1] [4]] [[2] [5]]  [[3] [6]]])

    print('----')
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = sum(x)
    z = reshape(y, (1, 1, 1))
    print(z)
    z.backward()
    print(y, x.grad)  # variable(15) [1 1 1 1 1 1]

    print('----')
    x = np.array([1, 2, 3])
    y = np.broadcast_to(x, (2, 3))
    print(y)  # [[1 2 3] [1 2 3]]

    print('----')
    x = Variable(np.array([1, 2, 3]))
    y = broadcast_to(x, (2, 3))
    y.backward()
    print(y)  # variable([[1 2 3] [1 2 3]])
    print(x.grad)  # 预期是 [2 2 2]

    print('----')
    x = Variable(np.array([[1, 2, 3], [1, 2, 3]]))
    y = sum_to(x, (3,))
    print(y)  # variable([3 6 9])
    y.backward()
    print(x.grad)  # variable([[1 1 1] [1 1 1]])

    print('--------')
    x = np.ones((1, 2))

    y = np.broadcast_to(x, (2, 2, 2))
    print(y)
    y = util_sum_to(y, (1, 2))
    print(y)

    print('----1-------')
    x = np.ones(24).reshape(2, 3, 4)
    y = np.sum(x, axis=(1, 2), keepdims=True)
    z = sum(Variable(x), (1, 2))
    print(y)  # [12. 12.]
    print(z)  # variable([12. 12.])
    print(z.shape)

    print('----2-------')
    x = Variable(np.array([1, 2, 3]))
    y = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    z = x + y
    print(z)  # variable([[2 4 6] [5 7 9]])
    z.backward()
    print(x.grad)  # [2 2 2]
    print(y.grad)  # [[1 1 1] [1 1 1]]
