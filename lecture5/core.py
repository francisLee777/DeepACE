import numpy as np


class Function:
    # 使用 * 写法把所有input_variable参数收集起来，打包成一个元组 args，这样则支持了任意数量输入变量，而不仅仅是一个
    def __call__(self, *input_variable):
        # 入参可能是非 Variable 类型，需要先转换成 Variable 类型
        input_variable = [as_variable(temp_x) for temp_x in input_variable]
        xs = [
            temp_x.value for temp_x in input_variable
        ]  # 从元组中取出所有变量对象，取出实际值，放到列表 xs 中
        ys = self.forward(*xs)  # 解包元组中的元素
        # 有些函数只返回一个输出（比如 ReLU），有些返回多个输出（比如 split），为了让后面逻辑统一处理成可迭代对象，这里强制转成 tuple。
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [
            Variable(as_array(temp_y)) for temp_y in ys
        ]  # 将计算结果封装成变量对象并返回
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，这样在反向传播时，可以沿着 output.creator 反查梯度来源

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        self.output_variable = (
            output_variable_list  # 保存输出变量，用于反向传播时计算梯度
        )
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


# start------------------------------------------------
#               加减乘除运算类
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

    def backward(self, gy):
        y = self.output_variable[0]
        gx = gy * (1 - y * y)
        return gx


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


# End----------------基础数学类 exp/sin/cos/tanh--------------


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

    def backward(self, dy):
        """
        执行反向传播。
        1. 通过 reshape 调整梯度形状。
        2. 使用广播机制将梯度广播回原始输入形状。
        """
        # 将梯度 reshape 为 "keep_dims=True" 时的形状
        dy_reshaped = reshape(dy, self.output_shape_kept)

        # 将梯度广播回原始形状
        dx = broadcast_to(dy_reshaped, self.origin_shape)
        return dx


def sum(input_x, axis=None, keep_dims=False):
    return Sum(axis, keep_dims)(input_x)


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


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:  # 偏置是可选值
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.input_variable
        db = None if b.value is None else sum_to(gy, b.shape)
        dx = matmul(gy, W.T)
        dW = matmul(x.T, gy)
        return dx, dW, db


def linear(x, W, b=None):
    return Linear()(x, W, b)


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


class Variable:
    __array_priority__ = 999

    def __init__(self, input_data, name=None):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError("{} is not supported".format(type(input_data)))
        self.name = name
        self.value = input_data
        self.grad = None  # 梯度 默认为 None
        self.creator = None  # 创建函数 默认为 None

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


if __name__ == "__main__":
    pass
