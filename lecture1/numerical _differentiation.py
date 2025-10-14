import numpy as np

from lecture1.function import Square, Exp
from lecture1.variable import Variable


# 数值微分, 传入函数和变量, 返回函数在这个变量上的微分
def numerical_differentiation(func, input_var, eps=1e-4):
    x0 = Variable(input_var.value - eps)
    x1 = Variable(input_var.value + eps)
    y0 = func(x0)
    y1 = func(x1)
    return (y1.value - y0.value) / (2 * eps)


if __name__ == '__main__':
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_differentiation(f, x)  # 输出 4.000000000004


    def f(input_var):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(input_var)))


    x = Variable(np.array(-1))
    dy = numerical_differentiation(f, x)  # 输出 -29.55622577500261
