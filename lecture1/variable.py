import numpy as np


class Variable:
    def __init__(self, input_data):
        self.value = input_data


# numpy 中，零维(dim)数据称为标量，一维数据称为向量，二维数据称为矩阵
if __name__ == '__main__':
    data = np.array(1.2)
    x = Variable(data)
    print("变量的值是", x.value)
