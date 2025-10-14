from lecture2.variable2 import *


class Function2:
    def __call__(self, *input_variable):
        # 利用数组，兼容多变量写法
        xs = [x.value for x in input_variable]  # 从变量对象中取出实际值
        ys = self.forward(*xs)  # 具体的计算在 forward 方法中，所有子类必须实现这个方法
        # 当 ys 不是可迭代对象时，用元组包裹起来，不然后面的迭代处理会出错
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable2(as_array(y)) for y in ys]  # 将计算结果封装成变量对象并返回
        for output in output_variable_list:
            output.creator = self  # 保存创建函数，用于反向传播时计算梯度

        self.input_variable = input_variable  # 保存输入变量，用于反向传播时计算梯度
        self.output_variable = output_variable_list  # 保存输出变量，用于反向传播时计算梯度
        # 如果返回值列表中只有一个元素，则返回第 1 个元素
        return output_variable_list if len(output_variable_list) > 1 else output_variable_list[0]

    # 所有子类必须实现这个方法
    def forward(self, *input_x):  # 先考虑只有一个输入变量
        raise NotImplementedError()

    def backward(self, input_dy):  # 先考虑只有一个输入变量
        raise NotImplementedError()


def as_array(input_data):
    if np.isscalar(input_data):
        return np.array(input_data)  # 转换成 np.array 类型
    return input_data


class Add(Function2):
    def forward(self, input1, input2):
        return input1 + input2

    def backward(self, input_dy):
        return 1 * input_dy, 1 * input_dy


def add(x0, x1):
    return Add()(x0, x1)


if __name__ == '__main__':
    x0 = Variable2(np.array(2))
    x1 = Variable2(np.array(3))
    y = add(x0, x1)
    print(y.value)

    y.backward()
