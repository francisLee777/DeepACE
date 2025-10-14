import numpy as np

from lecture1.variable import Variable


# 函数定义
class Function:
    # __call__ 是一个特殊方法，定义后, 能够 f = Function() 后直接调用 f(...)
    def __call__(self, input_variable):
        x = input_variable.value  # 从变量对象中取出实际值
        y = self.forward(x)  # 具体的计算在 forward 方法中，所有子类必须实现这个方法
        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        output_variable = Variable(y)  # 将计算结果封装成变量对象并返回
        return output_variable

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


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    f = Square()
    y = f(x)

    print(type(y))  # y 应该是"变量"类型
    print(type(f))  # f 应该是"Square"类型
    print(y.value)  # 输出 4.0

    x = Variable(np.array(-1))
    f = Square()
    g = Exp()
    h = g(f(x))
    print(h.value)  # 输出 2.718281828459045

    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.value)  # 输出 7.38905609893065

    db = C.backward(1)
    dc = B.backward(db)
    da = A.backward(dc)
    print(db, dc, da)
