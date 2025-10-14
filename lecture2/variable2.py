import numpy as np


class Variable2:
    def __init__(self, input_data):
        if input_data is not None and not isinstance(input_data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(input_data)))

        self.value = input_data
        self.grad = None  # 梯度 默认为 None
        self.creator = None  # 创建函数 默认为 None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.value)

        func = self.creator
        # 只有非用户输入的变量才会有 creator
        if func is not None:
            func.input_variable.grad = func.backward(self.grad)  # 调用变量创建函数的 backward 方法，计算梯度
            func.input_variable.backward()  # 继续递归调用，计算连接图中所有变量的梯度
