import numpy as np

from lecture2.core import Variable


# 更多的复杂函数

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


if __name__ == '__main__':
    x = Variable(np.array(3.0))
    y = Variable(np.array(1.0))

    z1 = goldstein(x, y)
    z1.backward()
    print(x.grad, y.grad)

    x.grad = 0
    y.grad = 0
    z2 = sphere(x, y)
    z2.backward()
    print(x.grad, y.grad)

    x.grad = 0
    y.grad = 0
    z3 = matyas(x, y)
    z3.backward()
    print(x.grad, y.grad)
