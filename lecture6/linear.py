import numpy as np

from lecture6.core import linear, sigmoid, mean_squared_error
from lecture6.layer import Layer
from lecture6.parameter import Parameter


class LinearWithInput(Layer):
    def __init__(self, input_size, output_size, need_bias=True, dtype=np.float32):
        super().__init__()
        I, O = input_size, output_size
        # 根据输入和输出的维度，初始化权重矩阵W，这里使用了Xavier初始化方法
        W_init = np.random.randn(I, O).astype(dtype) * np.sqrt(1.0 / I)
        self.W = Parameter(W_init, name="W")
        if need_bias:
            self.b = Parameter(np.zeros(O).astype(dtype), name="b")
        else:
            self.b = None

    def forward(self, x):
        return linear(x, self.W, self.b)


class Linear(Layer):
    # 不显式指定入参 input_size ， 在 forward 中根据输入动态确定
    def __init__(self, output_size, input_size=None, need_bias=True, dtype=np.float32):
        super().__init__()
        self.input_size, self.output_size, self.dtype = input_size, output_size, dtype

        self.W = Parameter(None, name="W")
        if self.input_size is not None:
            self._init_W()

        self.b = None
        if need_bias:
            self.b = Parameter(np.zeros(output_size).astype(dtype), name="b")

    def forward(self, inputs):
        # 如果之前没有指定输入维度，这里根据第一个输入的形状动态确定，并且初始化权重矩阵W
        if self.input_size is None:
            self.input_size = inputs.shape[1]
            self._init_W()

        return linear(inputs, self.W, self.b)

    def _init_W(self):
        I, O = self.input_size, self.output_size
        # 根据输入和输出的维度，初始化权重矩阵W，这里使用了Xavier初始化方法
        W_init = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1.0 / I)
        self.W.value = W_init


if __name__ == "__main__":
    x = np.random.randn(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.randn(100, 1)

    l1 = Linear(10)
    l2 = Linear(1)

    def predict(x):
        h = l1(x)
        y = sigmoid(h)
        return l2(y)

    lr = 0.1
    iters = 10000

    for i in range(iters):
        y_predict = predict(x)
        loss = mean_squared_error(y, y_predict)
        l1.clear_grads()
        l2.clear_grads()
        loss.backward()
        for l in [l1, l2]:
            for param in l.params():
                param.value -= lr * param.grad.value
        if i % 100 == 0:
            print(f"iter {i}, loss: {loss.value:.4f}")

    model = Layer()
    model.l1 = Linear(10)
    model.l2 = Linear(1)

    def predict(model, x):
        h = model.l1(x)
        y = sigmoid(h)
        return model.l2(y)

    model.clear_grads()
    # 省略训练代码

    for p in model.params():
        param.value -= lr * p.grad.value
