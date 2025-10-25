from lecture2.variable2 import *


class Function2:
    # 使用 * 写法把所有input_variable参数收集起来，打包成一个元组 args，这样则支持了任意数量输入变量，而不仅仅是一个
    def __call__(self, *input_variable):
        # 从元组中取出所有变量对象，取出实际值，放到列表 xs 中
        xs = [x.value for x in input_variable]  # 从变量对象中取出实际值
        ys = self.forward(*xs)  # 解包元组中的元素
        # 有些函数只返回一个输出（比如 ReLU），有些返回多个输出（比如 split），为了让后面逻辑统一处理成可迭代对象，这里强制转成 tuple。
        if not isinstance(ys, tuple):
            ys = (ys,)
        output_variable_list = [Variable3(as_array(y)) for y in ys]  # 将计算结果封装成变量对象并返回
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


def as_array(input_data):
    if np.isscalar(input_data):
        return np.array(input_data)  # 转换成 np.array 类型
    return input_data


class Add(Function2):
    def forward(self, input1, input2):
        return input1 + input2

    # backward 方法的返回值必须和 forward 方法的输入参数数量一致
    def backward(self, input_dy):
        return 1 * input_dy, 1 * input_dy


def add(x0, x1):
    return Add()(x0, x1)


class Multiplication(Function2):
    def forward(self, input1, input2):
        return input1 * input2

    def backward(self, input_dy):
        (input_x0, input_x1) = self.input_variable
        return input_dy * input_x1.value, input_dy * input_x0.value


def mul(input_x0, input_x1):
    input_x1 = as_array(input_x1)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    input_x0 = as_array(input_x0)
    return Multiplication()(input_x0, input_x1)


# 求平方函数，实现了 Function2 类
class Square(Function2):
    def forward(self, square_input):
        return square_input ** 2

    def backward(self, input_dy):
        # 注意：对于单输入函数，input_variable是一个只有一个元素的元组
        # (x, ) 把一个只包含一个元素的元组解包（unpack）成变量 x
        (x,) = self.input_variable
        return (2 * x.value * input_dy,)


# 平方函数的便捷接口
def square(input_variable):
    input_variable = as_array(input_variable)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Square()(input_variable)


# Exp 函数，实现了 Function 类
class Exp(Function2):
    def forward(self, input_x):
        return np.exp(input_x)

    def backward(self, input_dy):
        (x,) = self.input_variable
        return (input_dy * np.exp(x.value),)


# Exp 函数的便捷接口
def exp(input_variable):
    input_variable = as_array(input_variable)  # 转换成 np.array 类型，之后在 Function类中被转换为 Variable类型
    return Exp()(input_variable)


if __name__ == '__main__':
    x0 = Variable2(np.array(2))
    x1 = Variable2(np.array(3))
    z = add(square(x0), square(x1))
    z.backward()
    print(z.value)  # 2^2 + 3^2 = 13
    print(x0.grad)  # 4
    print(x1.grad)  # 6

    x2 = Variable2(np.array(1))
    k = square(x2)
    k.backward()
    print(x2.grad)

    x = Variable2(np.array(3.0))
    y = add(x, x)
    print('y', y.value)
    y.backward()
    print('x.grad', x.grad)

    # 第 2个计算（使用同一个 x进行其他计算）
    # 重置梯度！ 每次反向传播都获得独立的梯度结果（而不是累加），需要在每次反向传播前手动重置梯度
    x.grad = None
    z = add(add(x, x), x)
    z.backward()
    print(x.grad)

    print('----')

    x = Variable2(np.array(1.0))
    a = square(x)
    b = exp(a)
    c = exp(a)
    d = add(b, c)
    d.backward()
    print(x.grad)  # 16.30969097075427  正确的梯度应该是 10.873127495050205

    print('----')

    x = Variable3(np.array(1.0))
    a = square(x)
    b = exp(a)
    c = exp(a)
    d = add(b, c)
    d.backward()
    print(x.grad)  # 10.873127495050205

    print('----')

    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x.shape)  # (2, 3)
    print(len(x))  # 2
    print(x)  # [[1 2 3] [4 5 6]]

    print('----')

    x = Variable3(np.array([[1, 2, 3], [4, 5, 6]]))
    print(x.shape)  # (2, 3)
    print(len(x))  # 2
    print(x)  # variable([[1 2 3] [4 5 6]])
