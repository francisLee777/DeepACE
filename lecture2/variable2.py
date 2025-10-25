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
            output_grads = [y.grad for y in func.output_variable]
            gradList = func.backward(*output_grads)
            if not isinstance(gradList, tuple):
                gradList = (gradList,)
            for i, x in enumerate(func.input_variable):
                # 当一个变量被多个函数使用时，需要将多个函数的梯度累加起来
                if x.grad is None:
                    x.grad = gradList[i]
                else:
                    # 不能写成 x.grad += gradList[i]，因为 += 操作符在NumPy中会执行原地修改（in-place operation），这意味着它会直接修改 x.grad 指向的内存，而不是创建一个新的数组对象。
                    x.grad = x.grad + gradList[i]
            for x in func.input_variable:
                x.backward()  # 继续递归调用，计算连接图中所有变量的梯度


# 修复反向传播，后续遍历所有变量，保证梯度顺利地传播
class Variable3:
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

    # 运算符重载, 有循环引用问题，放在 core.py 中
    # def __mul__(self, other):
    #     return mul(self, other)
    #
    # def __add__(self, other):
    #     return add(self, other)

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
