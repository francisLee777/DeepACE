import numpy as np

from lecture8.core import Layer, Linear, sigmoid, tanh, Variable, Model, Embedding


# 简单 LSTM ，不带词嵌入
class SimpleLSTM(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = LSTM(hidden_size)
        self.fc = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, input_x):
        y = self.rnn(input_x)
        y = self.fc(y)
        return y


# 带有词嵌入的 LSTM
class LSTMWithEmbedding(Model):
    # out_size 一般是 vocab_size
    def __init__(self, hidden_size, out_size, embedding_size):
        super().__init__()
        self.embedding = Embedding(out_size, embedding_size)
        self.rnn = LSTM(hidden_size)
        self.fc = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, input_x):
        temp1 = self.embedding(input_x)
        y = self.rnn(temp1)
        y = self.fc(y)
        return y


class LSTM(Layer):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()

        self.h = None
        self.c = None

        H, I = hidden_size, input_size
        self.x2f = Linear(H, input_size=I)
        self.x2i = Linear(H, input_size=I)
        self.x2o = Linear(H, input_size=I)
        self.x2u = Linear(H, input_size=I)
        self.h2f = Linear(H, input_size=H, need_bias=False)
        self.h2i = Linear(H, input_size=H, need_bias=False)
        self.h2o = Linear(H, input_size=H, need_bias=False)
        self.h2u = Linear(H, input_size=H, need_bias=False)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = sigmoid(self.x2f(x))
            i = sigmoid(self.x2i(x))
            o = sigmoid(self.x2o(x))
            u = tanh(self.x2u(x))
        else:
            f = sigmoid(self.x2f(x) + self.h2f(self.h))
            i = sigmoid(self.x2i(x) + self.h2i(self.h))
            o = sigmoid(self.x2o(x) + self.h2o(self.h))
            u = tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            # 点积，不是矩阵乘法
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new


if __name__ == '__main__':
    lstm = LSTM(10)
    x = Variable(np.random.randn(20, 5))
    h = lstm(x)
    h.backward()
    print(h.shape, x.grad.shape)
