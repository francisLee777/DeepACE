import numpy as np


# 最终合并文件，防止循环引用问题


class Function:
    # 使用 * 写法把所有input_variable参数收集起来，打包成一个元组 args，这样则支持了任意数量输入变量，而不仅仅是一个
    def __call__(self, *input_variable):
        # 入参可能是非 Variable 类型，需要先转换成 Variable 类型
        input_variable = [as_variable(x) for x in input_variable]
        xs = [x.value for x in input_variable]  # 从元组中取出所有变量对象，取出实际值，放到列表 xs 中
        ys = self.forward(*xs)  # 解包元组中的元素
        # 有些函数只返回一个输出（比如 ReLU），有些返回多个输出（比如 split），为了让后面逻辑统一处理成可迭代对象，这里强制转成 tuple。
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable(as_array(y)) for y in ys]  # 将计算结果封装成变量对象并返回
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，这样在反向传播时，可以沿着 output.creator 反查梯度来源

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        self.output_variable = output_variable_list  # 保存输出变量，用于反向传播时计算梯度
        # 如果返回值列表中只有一个元素，则返回第 1 个元素。
        # 这种处理方式的优点是符合人类直觉，但缺点是返回值类型不固定，需要调用者根据实际情况决定如何取值，y, = Square(x) 单输出时加逗号解包  y1, y2 = split(x) # 多输出时正常解包
        # 作为教学项目比较合理，但工业级框架一般固定为返回一个 tuple/tensor, 这样可以统一处理单输出和多输出的情况
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

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


class Add(Function):
    def forward(self, input1, input2):
        return input1 + input2

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致
    def backward(self, input_dy):
        return 1 * input_dy, 1 * input_dy


def add(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Add()(x0, x1)


class Multiplication(Function):
    def forward(self, input1, input2):
        return input1 * input2

    def backward(self, input_dy):
        (input_x0, input_x1) = self.input_variable
        return input_dy * input_x1.value, input_dy * input_x0.value


def mul(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Multiplication()(x0, x1)


class Sub(Function):
    def forward(self, input1, input2):
        return input1 - input2

    def backward(self, input_dy):
        return 1 * input_dy, -1 * input_dy


def sub(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Sub()(x0, x1)


class Pow(Function):
    def __init__(self, power):
        self.power = power

    def forward(self, input_x):
        return input_x ** self.power

    def backward(self, input_dy):
        (x,) = self.input_variable
        return self.power * (x.value ** (self.power - 1)) * input_dy


def pow(x, power):
    x = as_array(x)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Pow(power)(x)


class Div(Function):
    def forward(self, input1, input2):
        return input1 / input2

    def backward(self, input_dy):
        (input_x0, input_x1) = self.input_variable
        return input_dy / input_x1.value, -input_dy * input_x0.value / (input_x1.value ** 2)


def div(x0, x1):
    x1 = as_array(x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    x0 = as_array(x0)
    return Div()(x0, x1)


class Neg(Function):
    def forward(self, input_x):
        return -input_x

    def backward(self, input_dy):
        return -input_dy


def neg(x):
    x = as_array(x)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Neg()(x)


class Abs(Function):
    def forward(self, input_x):
        return np.abs(input_x)

    def backward(self, input_dy):
        (input_x,) = self.input_variable
        return input_dy * np.sign(input_x.value)


def abs(x):
    x = as_array(x)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Abs()(x)


# 求平方函数，实现了 Function2 类
class Square(Function):
    def forward(self, square_input):
        return square_input ** 2

    def backward(self, input_dy):
        # 注意：对于单输入函数，input_variable是一个只有一个元素的元组
        # (x, ) 把一个只包含一个元素的元组解包（unpack）成变量 x
        (x,) = self.input_variable
        return (2 * x.value * input_dy,)


# 平方函数的便捷接口
def square(input_variable):
    return Square()(input_variable)


# Exp 函数，实现了 Function 类
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_dy):
        (x,) = self.input_variable
        return (input_dy * np.exp(x.value),)


# Exp 函数的便捷接口
def exp(input_variable):
    return Exp()(input_variable)


class Variable:
    __array_priority__ = 999

    def __init__(self, input_data):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(input_data)))

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
            return 'variable(None)'
        p = str(self.value).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

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

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.value)

        # 创建一个列表来存储需要处理的函数和梯度对
        funcs = []
        visited = set()  # 用于跟踪已访问的函数，避免重复处理

        # 后序遍历收集所有函数
        def add_func(temp_func):
            if temp_func not in visited:
                # 先添加输入变量的创建函数
                visited.add(temp_func)
                # 把输入变量的所有创建函数也添加到列表中
                for temp_x in temp_func.input_variable:
                    if temp_x.creator is not None:
                        add_func(temp_x.creator)
                # 再添加当前函数
                funcs.append(temp_func)

        # 如果当前变量有创建函数，开始收集
        if self.creator is not None:
            add_func(self.creator)

        # 按照后序遍历的逆序（从输出到输入）处理每个函数
        for f in funcs[::-1]:
            # 计算当前函数的梯度
            output_grads = [y.grad for y in f.output_variable]
            grads = f.backward(*output_grads)
            if not isinstance(grads, tuple):
                grads = (grads,)

            # 将梯度传递给输入变量
            for i, x in enumerate(f.input_variable):
                if x.grad is None:
                    x.grad = grads[i]
                else:
                    # 不能写成 temp_x.grad += grads[i]，否则在 Python 的语义中，就地修改原有对象，如果其他节点仍然在依赖这个 temp_x.grad, 会被污染数据。
                    x.grad = x.grad + grads[i]


if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(x + y)  # variable([[2 4 6] [8 10 12]])
    print(x * y)  # variable([[1 4 9] [16 25 36]])

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    print(x + np.array([[1, 2, 3], [4, 5, 6]]))  # variable([[2 4 6] [8 10 12]])
    print(x + 1)  # variable([[2 3 4] [5 6 7]])

    print(np.array([[1, 2, 3], [4, 5, 6]]) + x)  # variable([[2 3 4] [5 6 7]])
    print(np.array(1) + x)  # variable([[2 3 4] [5 6 7]])

    x = Variable(np.array(1))
    y = Variable(np.array(2))
    print(np.array(1) + x)
    # print((x + 1) * (1 + y))
