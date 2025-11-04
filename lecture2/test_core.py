import numpy as np

from lecture2.core import square, Variable

x2 = Variable(np.array(1))
k = square(x2)
k.backward()
print(x2.grad)  # 和
