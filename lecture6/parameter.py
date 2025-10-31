import numpy as np

from lecture6.core import Variable


# 变量类，继承Variable类
class Parameter(Variable):
    pass


if __name__ == "__main__":
    x = Variable(np.array(1.0))
    p = Parameter(np.array(2.0))
    y = x * p  # 有效计算，继承关系。
    print(isinstance(p, Parameter))  # True
    print(isinstance(x, Parameter))  # False
    # False, 因为在现有代码中， mul 乘法函数会执行 __call__ 方法，内部将值转换成了Variable实例
    print(isinstance(y, Parameter))
