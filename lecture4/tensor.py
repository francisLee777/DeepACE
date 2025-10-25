import numpy as np

from lecture3.core import Variable, neg, abs


def temp_fun(x, y):
    return pow(x + 1, 2) * neg(y) - abs(x - y)


if __name__ == '__main__':
    # 2维矩阵
    x = Variable(np.array([[1, 2, 3], [3, 4, 5]]))
    y = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    z = temp_fun(x, y)
    z.backward()
    print("z:", z)  # variable([[  -4  -18  -48] [ -65 -126 -217]])
    print("x.grad:", x.grad)  # [[ -4 -12 -24] [-31 -49 -71]]
    print("y.grad:", y.grad)  # [[ -4  -9 -16] [-17 -26 -37]]

    # 3维矩阵
    x = Variable(np.array([[[1, 2], [3, 4]], [[1, 2], [4, 5]]]))
    y = Variable(np.array([[[1, 2], [4, 5]], [[1, 2], [4, 5]]]))
    z = temp_fun(x, y)
    z.backward()
    print("z:", z)  # z: variable([[[  -4  -18] [ -65 -126]]  [[  -4  -18]  [-100 -180]]])
    print("x.grad:", x.grad)  # [[[ -4 -12] [-31 -49]][[ -4 -12] [-40 -60]]]
    print("y.grad:", y.grad)  # [[[ -4  -9][-17 -26]][[ -4  -9] [-25 -36]]]
