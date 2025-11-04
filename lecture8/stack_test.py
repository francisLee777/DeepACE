import numpy as np

from lecture8.core import Variable, stack

# 示例1：使用全局 stack 函数堆叠多个 Variable
v1 = Variable(np.array([[1, 2], [3, 4]]))
v2 = Variable(np.array([[5, 6], [7, 8]]))
v3 = Variable(np.array([[9, 10], [11, 12]]))
stacked = stack([v1, v2, v3], axis=0)
print(stacked.shape)  # 输出 (3, 2, 2)
print(stacked)
stacked.backward()
print(v1.grad)

# 示例2：使用实例方法堆叠两个 Variable
v1.clear_grad()
v2.clear_grad()
v3.clear_grad()
stacked.clear_grad()
stacked2 = stacked.stack(stacked, axis=0)
print(stacked2.shape)  # 输出 (2, 3, 2, 2)
stacked2.backward()
print(v1.grad, stacked.grad)
