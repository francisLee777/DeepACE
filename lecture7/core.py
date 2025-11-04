import math

import numpy as np

from final_ACE_framework.graph_util import plot_dot_graph


# Start -----------------------------------------------
#           核心基础 Variable 和 Function
# -----------------------------------------------------
class Function:
    # 使用 * 写法把所有input_variable参数收集起来，打包成一个元组 args，这样则支持了任意数量输入变量，而不仅仅是一个
    def __call__(self, *input_variable):
        # 入参可能是非 Variable 类型，需要先转换成 Variable 类型
        input_variable = [as_variable(temp_x) for temp_x in input_variable]
        # 从元组中取出所有变量对象，取出实际值，放到列表 xs 中
        xs = [temp_x.value for temp_x in input_variable]
        ys = self.forward(*xs)  # 解包元组中的元素
        # 有些函数只返回一个输出（比如 ReLU），有些返回多个输出（比如 split），为了让后面逻辑统一处理成可迭代对象，这里强制转成 tuple。
        if not isinstance(ys, tuple):
            ys = (ys,)
            # 将计算结果封装成变量对象并返回
        output_variable_list = [Variable(as_array(temp_y)) for temp_y in ys]
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，这样在反向传播时，可以沿着 output.creator 反查梯度来源

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        # 保存输出变量，用于反向传播时计算梯度
        self.output_variable = output_variable_list
        # 如果返回值列表中只有一个元素，则返回第 1 个元素。
        # 这种处理方式的优点是符合人类直觉，但缺点是返回值类型不固定，需要调用者根据实际情况决定如何取值，y, = Square(x) 单输出时加逗号解包  y1, y2 = split(x) # 多输出时正常解包
        # 作为教学项目比较合理，但工业级框架一般固定为返回一个 tuple/tensor, 这样可以统一处理单输出和多输出的情况
        return (
            output_variable_list
            if len(output_variable_list) > 1
            else output_variable_list[0]
        )

    # 所有子类必须实现这个方法
    def forward(self, *input_x):
        raise NotImplementedError()

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致
    def backward(self, input_dy):
        raise NotImplementedError()


# 将 np.array 转换成 Variable 类型
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(input_data):
    if np.isscalar(input_data):
        return np.array(input_data)  # 转换成 np.array 类型
    return input_data


class Variable:
    __array_priority__ = 999

    def __init__(self, input_data, name=None):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError("{} is not supported".format(type(input_data)))
        self.name = name
        self.value = input_data
        self.grad = None  # 梯度 默认为 None
        self.creator = None  # 创建函数 默认为 None

    def clear_grad(self):
        self.grad = None

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @property
    def dtype(self):
        return self.value.dtype

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        if self.value is None:
            return "variable(None)"
        p = str(self.value).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    # 运算符重载
    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __getitem__(self, item):
        return get_item(self, item)

    def __rsub__(self, other):
        return sub(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __neg__(self):
        return neg(self)

    def __abs__(self):
        return abs(self)

    def dot(self, other):
        return matmul(self, other)

    def matmul(self, other):
        return matmul(self, other)

    def reshape(self, *shape):
        # 如果入参 shape 是一个元组或列表，且只有一个元素，
        # 则将该元素解包（unpack）为 shape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *input_axes):
        if len(input_axes) == 0:
            input_axes = None
        elif len(input_axes) == 1:
            # 兼容一下，当只有一个元素时，解包为 axes
            if isinstance(input_axes[0], (tuple, list)) or input_axes[0] is None:
                input_axes = input_axes[0]
        return transpose(self, input_axes)

    @property
    def T(self):
        return self.transpose()  # 相当于逆序，不需要参数

    def sum(self, axis=None, keepdims=False):
        return sum(self, axis, keepdims)

    def backward(self):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.value))

        # 创建一个列表来存储需要处理的函数和梯度对
        funcs = []
        visited = set()  # 用于跟踪已访问的函数，避免重复处理

        # 后序遍历收集所有函数
        def add_func(temp_func):
            if temp_func not in visited:
                # 先添加输入变量的创建函数
                visited.add(temp_func)
                # 把输入变量的所有创建函数也添加到列表中
                for temp_xx in temp_func.input_variable:
                    if temp_xx.creator is not None:
                        add_func(temp_xx.creator)
                # 再添加当前函数
                funcs.append(temp_func)

        # 如果当前变量有创建函数，开始收集
        if self.creator is not None:
            add_func(self.creator)

        # 按照后序遍历的逆序（从输出到输入）处理每个函数
        for f in funcs[::-1]:
            # 计算当前函数的梯度
            output_grads = [temp_y.grad for temp_y in f.output_variable]
            grads = f.backward(*output_grads)
            if not isinstance(grads, tuple):
                grads = (grads,)

            # 将梯度传递给输入变量
            for i, temp_x in enumerate(f.input_variable):
                if temp_x.grad is None:
                    temp_x.grad = grads[i]
                else:
                    # 不能写成 temp_x.grad += grads[i]，否则在 Python 的语义中，就地修改原有对象，如果其他节点仍然在依赖这个 temp_x.grad, 会被污染数据。
                    temp_x.grad = temp_x.grad + grads[i]


# End ------------------------------------------------
#             Variable 和 Function
# -----------------------------------------------------


# start------------------------------------------------
#               加减乘除等基础运算类
# -----------------------------------------------------
class Add(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        # 由于是 np.array 类型，所以可以直接进行元素级别的加法，np 会自动广播
        return input1 + input2

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致
    def backward(self, input_dy):
        input_dy1, input_dy2 = input_dy, input_dy
        # 处理广播情况
        if self.input1_shape != self.input2_shape:
            input_dy1 = sum_to(input_dy1, self.input1_shape)
            input_dy2 = sum_to(input_dy2, self.input2_shape)
        return 1 * input_dy1, 1 * input_dy2


def add(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Add()(x0, x1)


class Sub(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        return input1 - input2

    def backward(self, input_dy):
        # 处理广播
        dy1, dy2 = input_dy, - input_dy
        if self.input1_shape != self.input2_shape:
            dy1 = sum_to(dy1, self.input1_shape)
            dy2 = sum_to(dy2, self.input2_shape)
        return dy1, dy2


def sub(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Sub()(x0, x1)


class Multiplication(Function):
    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        return input1 * input2

    def backward(self, input_dy):
        (input_x0, input_x1) = self.input_variable
        # 处理广播
        dy1, dy2 = input_dy * input_x1.value, input_dy * input_x0.value
        if self.input1_shape != self.input2_shape:
            dy1 = sum_to(dy1, self.input1_shape)
            dy2 = sum_to(dy2, self.input2_shape)
        return dy1, dy2


def mul(input_x0, input_x1):
    input_x1 = as_array(
        input_x1
    )  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    input_x0 = as_array(input_x0)
    return Multiplication()(input_x0, input_x1)


class Pow(Function):
    def __init__(self, power):
        self.power = power

    def forward(self, input_x):
        return input_x ** self.power

    def backward(self, input_dy):
        (input_x,) = self.input_variable
        return self.power * (input_x ** (self.power - 1)) * input_dy


def pow(input_x, power):
    input_x = as_array(
        input_x
    )  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Pow(power)(input_x)


class Div(Function):

    def __init__(self):
        self.input1_shape = None
        self.input2_shape = None

    def forward(self, input1, input2):
        self.input1_shape, self.input2_shape = input1.shape, input2.shape
        return input1 / input2

    def backward(self, input_dy):
        input_x0, input_x1 = self.input_variable
        dy1, dy2 = input_dy / input_x1, -input_dy * input_x0 / (
                input_x1 ** 2
        )
        # 处理广播
        if self.input1_shape != self.input2_shape:
            dy1 = sum_to(dy1, self.input1_shape)
            dy2 = sum_to(dy2, self.input2_shape)
        return dy1, dy2


def div(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Div()(x0, x1)


class Neg(Function):
    def forward(self, input_x):
        return -input_x

    def backward(self, input_dy):
        return -input_dy


def neg(input_x):
    input_x = as_array(
        input_x
    )  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Neg()(input_x)


class Abs(Function):
    def forward(self, input_x):
        return np.abs(input_x)

    def backward(self, input_dy):
        (input_x,) = self.input_variable
        return input_dy * np.sign(input_x.value)


def abs(input_x):
    input_x = as_array(
        input_x
    )  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Abs()(input_x)


# 求平方函数，实现了 Function2 类
class Square(Function):
    def forward(self, square_input):
        return square_input ** 2

    def backward(self, input_dy):
        # 注意：对于单输入函数，input_variable是一个只有一个元素的元组
        # (x, ) 把一个只包含一个元素的元组解包（unpack）成变量 x
        (x,) = self.input_variable
        return 2 * x.value * input_dy


# 平方函数的便捷接口
def square(input_variable):
    input_variable = as_array(
        input_variable
    )  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Square()(input_variable)


# End ------------------------------------------------
#               加减乘除等基础运算类
# -----------------------------------------------------


# Start ------------------------------------------------
#               张量操作类[形状有关]
# -----------------------------------------------------
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


class Reshape(Function):
    def __init__(self, target_shape):
        self.origin_shape = None  # 先声明
        self.target_shape = target_shape

    def forward(self, x):
        self.origin_shape = x.shape
        return np.reshape(x, self.target_shape)

    def backward(self, dy):
        # 这里要使用自身的reshape，而不是np.reshape，因为输入输出都是 Variable 类型
        return reshape(dy, self.origin_shape)  # 反向传播时，需要将dy的形状恢复到x的形状


def reshape(input_x, shape):
    if input_x.shape == shape:
        return as_variable(input_x)
    return Reshape(shape)(as_array(input_x))


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


class Sum(Function):
    """
    沿指定轴计算张量的元素总和。
    """

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.output_shape_kept = None
        self.origin_shape = None

    def forward(self, input_x):
        """
        执行前向传播。
        1. 保存输入形状 `self.origin_shape`，这对于反向传播至关重要。
        2. 计算并保存 `self.output_shape_kept`，即使 `keepdims=False`，输出的形状。
        3. 使用 np.sum 执行实际的求和操作。
        """
        self.origin_shape = input_x.shape
        # 计算 "keepdims=True" 时的输出形状
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
        y = np.sum(input_x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, dy):
        """
        执行反向传播。
        1. 通过 reshape 调整梯度形状。
        2. 使用广播机制将梯度广播回原始输入形状。
        """
        # 将梯度 reshape 为 "keepdims=True" 时的形状
        dy_reshaped = reshape(dy, self.output_shape_kept)

        # 将梯度广播回原始形状
        dx = broadcast_to(dy_reshaped, self.origin_shape)
        return dx


def sum(input_x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(input_x)


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


# End ------------------------------------------------
#               张量操作类[形状有关]
# -----------------------------------------------------


# start------------------------------------------------
#               基础数学类 exp/sin/cos/tanh
# -----------------------------------------------------


# Exp 函数，实现了 Function 类
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_dy):
        (out_dy,) = self.output_variable
        return input_dy * out_dy


# Exp 函数的便捷接口
def exp(input_variable):
    input_variable = as_array(
        input_variable
    )  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Exp()(input_variable)


class Sin(Function):
    def forward(self, input_x):
        y = np.sin(input_x)
        return y

    def backward(self, dy):
        (x,) = self.input_variable
        dx = dy * cos(x)
        return dx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, input_x):
        y = np.cos(input_x)
        return y

    def backward(self, dy):
        (x,) = self.input_variable
        dx = dy * -sin(x)
        return dx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, input_x):
        y = np.tanh(input_x)
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = dy * (1 - y * y)
        return dx


def tanh(x):
    return Tanh()(x)


class Log(Function):
    def forward(self, input_x):
        y = np.log(input_x)
        return y

    def backward(self, dy):
        (x,) = self.input_variable
        dx = dy / x
        return dx


def log(x):
    return Log()(x)


# End ------------------------------------------------
#            基础数学类 exp/sin/cos/tanh
# -----------------------------------------------------


# Start -----------------------------------------------
#               max/min/clip
# -----------------------------------------------------


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


#   End -----------------------------------------------
#               max/min/clip
# -----------------------------------------------------


# Start -----------------------------------------------
#               矩阵乘法和线性回归
# -----------------------------------------------------
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


# 作为 Function 实现，还有实现为 Layer 的版本
class LinearFunction(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:  # 偏置是可选值
            y += b
        return y

    def backward(self, dy):
        x, W, b = self.input_variable
        db = None if b.value is None else sum_to(dy, b.shape)
        dx = matmul(dy, W.T)
        dW = matmul(x.T, dy)
        return dx, dW, db


def linear(x, W, b=None):
    return LinearFunction()(x, W, b)


# End -----------------------------------------------
#               矩阵乘法和线性回归
# -----------------------------------------------------


# Start -----------------------------------------------
#             激活函数和损失函数 sigmoid 和 softmax
# -----------------------------------------------------
class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        # y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = dy * y * (1 - y)
        return dx


def sigmoid(x):
    return Sigmoid()(x)


def accuracy(y, t):
    # 输入需要是 one-hot 编码
    y, t = as_variable(y), as_variable(t)
    # 预测值中概率最大的类别，构成与标签相同的形状
    pred = y.value.argmax(axis=1).reshape(t.shape)
    result = pred == t.value  # 预测值与标签值相等的位置为True，否则为False
    acc = result.mean()  # 计算准确率，即正确预测的样本数占总样本数的比例
    return Variable(as_array(acc))


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
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, dy):
        y = self.output_variable[0]
        dx = y * dy
        sum_dx = dx.sum(axis=self.axis, keepdims=True)
        dx -= y * sum_dx
        return dx


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, dy):
        # 反向传播： dL/dx = (y - one_hot(t)) / N
        # 拿到保存的值
        x, t = self.input_variable
        N, _ = x.shape

        dy *= 1 / N
        y = softmax(x)

        # 构造 one-hot
        one_hot = np.zeros_like(y, dtype=t.dtype)
        one_hot[np.arange(N), t.value] = 1
        # softmax + crossentropy 的合成梯度
        y = (y - one_hot) * dy
        # 这个Node是指 t 无需梯度，因为t是标签，不是中间变量。如果没有None的话导致框架后续逻辑不兼容
        return y, None


# 使用另一种计算方式。是数学上最稳定的表达形式，避免任何溢出或 underflow。
def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def softmax(x, axis=1):
    return Softmax(axis)(x)


class MeanSquaredError(Function):
    def forward(self, input_x0, input_x1):
        diff = input_x0 - input_x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, dy):
        x0, x1 = self.input_variable
        diff = x0 - x1
        dx0 = dy * diff * (2.0 / len(diff))
        dx1 = -dx0
        return dx0, dx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


#  End  -----------------------------------------------
#             激活函数和损失函数
# -----------------------------------------------------

# Start ------------------------------------------------
#             Parameter 、 Layer 和 Model
# -----------------------------------------------------


# 变量类，继承Variable类
class Parameter(Variable):
    pass


class Layer:
    def __init__(self):
        self._params_name = set()  # set 是集合，无序且元素唯一

    def __setattr__(self, name, value):
        # 只搜集Parameter，不搜集Variable
        if isinstance(value, (Parameter, Layer)):
            self._params_name.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # tuple 不可变，转换成 list 类型
        self.inputs, self.outputs = list(inputs), list(outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params_name:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    # 清除所有参数的梯度
    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params_name:
            # __dict__ 是类的属性字典，包含了类的所有属性，包括实例属性和类属性
            obj = self.__dict__[name]

            # key 的设计是支持嵌套的，例如：layer1/W 表示 layer1 层的权重参数 W
            key = parent_key + '/' + name if parent_key else name
            # 如果是 Layer 类型，递归调用 _flatten_params 方法，最终都将参数加入到 params_dict 中
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_params(self, file_path="params.npz"):
        params_dict = {}
        self._flatten_params(params_dict)
        # params_dict 中的对象是 Parameter 类型，要转换成 numpy 数组类型
        for key, param in params_dict.items():
            params_dict[key] = param.value
        np.savez_compressed(file_path, **params_dict)

    def load_params(self, file_path="params.npz"):
        data = np.load(file_path, allow_pickle=True)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.value = data[key]


class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)


class Linear(Layer):
    # 不显式指定入参 input_size ， 在 forward 中根据输入动态确定
    def __init__(self, output_size, input_size=None, need_bias=True, dtype=np.float32):
        super().__init__()
        self.input_size, self.output_size, self.dtype = input_size, output_size, dtype

        self.W = Parameter(None, name="W")
        if self.input_size is not None:
            self._init_W()

        self.b = None
        if need_bias:
            self.b = Parameter(np.zeros(output_size).astype(dtype), name="b")

    def forward(self, inputs):
        # 如果之前没有指定输入维度，这里根据第一个输入的形状动态确定，并且初始化权重矩阵W
        if self.input_size is None:
            self.input_size = inputs.shape[1]
            self._init_W()

        return linear(inputs, self.W, self.b)

    def _init_W(self):
        I, O = self.input_size, self.output_size
        # 根据输入和输出的维度，初始化权重矩阵W，这里使用了Xavier初始化方法
        W_init = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1.0 / I)
        self.W.value = W_init


class MLP(Model):
    # fc_output_sizes 全连接层的输出维度列表[不需要指定输入维度，自动推断]
    def __init__(self, fc_output_sizes, activation=sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, "layer" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        # 前向传播：依次线性 + 激活（最后一层不加激活）
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        # 最后一层不激活，直接计算并返回
        return self.layers[-1](x)


# End ------------------------------------------------
#             Parameter 、 Layer 和 Model
# -----------------------------------------------------


# Start ------------------------------------------------
#                       优化器
# -----------------------------------------------------
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


# End ------------------------------------------------
#                       优化器
# -----------------------------------------------------


# Start -----------------------------------------------
#                       数据集相关
# -----------------------------------------------------


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


def get_example_data():
    num_data, num_class, input_dim = 1000, 3, 2
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


# End -----------------------------------------------
#                       数据集相关
# -----------------------------------------------------


if __name__ == "__main__":
    pass
