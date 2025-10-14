import numpy as np

from lecture2.variable import Variable


class Function:
    # __call__ 是一个特殊方法，定义后, 能够 f = Function() 后直接调用 f(...)
    def __call__(self, *input_variable):
        # 利用数组，兼容多变量写法
        xs = [x.input_variable for x in input_variable]  # 从变量对象中取出实际值
        ys = self.forward(*xs)  # 具体的计算在 forward 方法中，所有子类必须实现这个方法
        if not isinstance(ys, tuple):  # 对非元组情况的额外处理
            ys = (ys,)
        output_variable_list = [Variable(as_array(y)) for y in ys]  # 将计算结果封装成变量对象并返回
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，用于反向传播时计算梯度

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        self.output_variable = output_variable_list  # 保存输出变量，用于反向传播时计算梯度
        # 如果返回值列表中只有一个元素，则返回第 1 个元素
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

    # 所有子类必须实现这个方法
    def forward(self, input_x):  # 先考虑只有一个输入变量
        raise NotImplementedError()

    def backward(self, input_dy):  # 先考虑只有一个输入变量
        raise NotImplementedError()


# 求平方函数，实现了 Function 类
class Square(Function):
    def forward(self, input_x):
        return input_x ** 2

    def backward(self, input_dy):
        return 2 * self.input_variable.value * input_dy


# Exp 函数，实现了 Function 类
class Exp(Function):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_dy):
        return input_dy * np.exp(self.input_variable.value)


def square(input_variable):
    return Square()(input_variable)


def exp(input_variable):
    return Exp()(input_variable)


def as_array(input_data):
    if np.isscalar(input_data):
        return np.array(input_data)  # 转换成 np.array 类型
    return input_data


if __name__ == '__main__':
    x = np.array(1)  # numpy.ndarray 类型
    print(type(x ** 2))  # 输出 numpy.float64 类型

    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(-1))
    a = A(x)
    b = B(a)
    y = C(b)

    print(y.creator)
    print(y.creator.input_variable.value)
    print(y.creator.input_variable.creator)
    print(y.creator.input_variable.creator.input_variable.value)

    # 进行反向传播
    y.grad = np.array(1)  # 默认梯度的原始值是1
    C = y.creator  # 获取最后一个变量的创建函数
    b = C.input_variable  # 获取创建函数的输入变量
    b.grad = C.backward(y.grad)  # 调用 backward 方法，计算出输入变量对应的梯度
    # 重复上述过程，计算出所有变量的梯度
    B = b.creator
    x = B.input_variable
    x.grad = B.backward(b.grad)
    A = x.creator
    x = A.input_variable
    x.grad = A.backward(a.grad)
    print(y.grad, b.grad, a.grad, x.grad)

    print('-----------')
    y.grad = np.array(1)  # 默认梯度的原始值是1

    # 通过 creator 方式建立了反向的连接图，并进行反向传播计算梯度
    b.grad = y.creator.backward(y.grad)
    a.grad = y.creator.input_variable.creator.backward(b.grad)
    x.grad = y.creator.input_variable.creator.input_variable.creator.backward(a.grad)
    print(b.grad, a.grad, x.grad)

    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(1))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1)  # 默认梯度的原始值是1
    y.backward()  # 调用 backward 方法，内部递归计算图中所有变量的梯度
    print(x.grad)  # 输出为 29.556224395722598
