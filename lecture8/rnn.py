import numpy as np

from lecture8.core import Linear, tanh, Model, Layer, Variable


class RNN(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.x_w = Linear(hidden_size)
        # 这里不使用偏置的原因是，在 x_w 中已经包括了偏置
        self.h_prev_w = Linear(hidden_size, need_bias=False)
        self.h_cur = None  # 当前步骤的隐藏状态，初始为None，每次前向传播后更新

    def reset_state(self):
        self.h_cur = None

    def forward(self, x):
        if self.h_cur is None:
            # 第一个时间步，没有隐藏状态，所以直接使用输入计算隐藏状态
            h_new = tanh(self.x_w(x))
        else:
            h_new = tanh(self.x_w(x) + self.h_prev_w(self.h_cur))
        self.h_cur = h_new
        # 输出的维度是 x_batch_size, hidden_size
        return h_new


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = RNN(hidden_size)
        self.fc = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, input_x):
        h = self.rnn(input_x)
        y = self.fc(h)
        return y


if __name__ == "__main__":
    rnn = RNN(5)
    x1 = Variable(np.random.randn(2, 16))
    h = rnn(x1)
    print(h)
    x2 = Variable(np.random.randn(2, 16))
    h = rnn(x2)
    print(h)
    x3 = Variable(np.random.randn(2, 16))
    h = rnn(x3)
    print(h)
