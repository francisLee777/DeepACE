import numpy as np

from lecture8.core import Linear, tanh, Layer, stack, Variable, softmax, logsumexp, Function


class TimeRNN(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.x_w = Linear(hidden_size)
        self.h_prev_w = Linear(hidden_size, need_bias=False)
        self.h_cur = None

    def reset_state(self, h=None):
        self.h_cur = h

    def forward(self, x):
        # 输入是应该是 (batch_size, time_steps, input_size) 或者 不做词嵌入时是 (batch_size, time_steps)
        if x.ndim == 2:
            # 兼容无 input_size 维度的情况
            x = x[:, :, None]

        batch_size, time_steps, _ = x.shape
        # t 个 (batch_size, hidden_size), 需要 stack 起来
        h_list = [self._step(x[:, t, :]) for t in range(time_steps)]
        return stack(h_list, axis=1)  # 使用 Variable 的 stack 方法 , 输出 (batch_size, time_steps, hidden_size)

    def _step(self, x_t):
        if self.h_cur is None:
            h_new = tanh(self.x_w(x_t))
        else:
            h_new = tanh(self.x_w(x_t) + self.h_prev_w(self.h_cur))
        self.h_cur = h_new
        return h_new


class TimeLinear(Layer):
    def __init__(self, output_size, need_bias=True):
        super().__init__()
        self.fc = Linear(output_size, need_bias=need_bias)

    def forward(self, x):
        """
        x: (batch_size, time_steps, input_size) 或 (batch_size, input_size)
        return: (batch_size, time_steps, output_size)
        """
        if x.ndim == 3:
            batch_size, time_steps, input_size = x.shape
            x_flat = x.reshape(batch_size * time_steps, input_size)
            y_flat = self.fc(x_flat)
            y = y_flat.reshape(batch_size, time_steps, -1)
            return y
        elif x.ndim == 2:
            # 没有时间步的单步输入
            return self.fc(x)[:, None, :]  # 保留时间维以统一结构
        else:
            raise ValueError(f"输入维度必须为2或3，当前是 {x.ndim}")


class TimeSimpleRNN(Layer):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.rnn = TimeRNN(hidden_size)
        self.fc = TimeLinear(output_size)

    def reset_state(self, h=None):
        self.rnn.reset_state(h)

    def forward(self, x):
        h_seq = self.rnn(x)
        y_seq = self.fc(h_seq)
        return y_seq


class TimeSoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        """
        x: (N, T, C)
        t: (N, T)  值是 C
        """
        N, T, C = x.shape

        # Flatten 到 (N*T, C)
        x_flat = x.reshape(N * T, C)
        t_flat = t.reshape(N * T)

        # 防止标签超界
        assert t_flat.ndim == 1 and np.all((t_flat >= 0) & (t_flat < C))

        # logsumexp for numerical stability
        log_z = logsumexp(x_flat, axis=1)
        log_p = x_flat - log_z
        log_p = log_p[np.arange(N * T), t_flat]

        # 求平均损失
        loss = -log_p.mean().astype(np.float32)
        return loss

    def backward(self, dy):
        x, t = self.input_variable  # 从保存的 Variable 里取出
        N, T, C = x.shape

        dy *= 1 / (N * T)

        # softmax
        y = softmax(x.reshape(N * T, C))

        # one-hot
        t_flat = t.value.reshape(N * T)
        one_hot = np.zeros_like(y, dtype=t.dtype)
        one_hot[np.arange(N * T), t_flat] = 1

        # 计算梯度
        gx = (y - one_hot) * dy
        gx = gx.reshape(N, T, C)

        # 标签无梯度
        return gx, None


def time_softmax_cross_entropy(x, t):
    return TimeSoftmaxCrossEntropy()(x, t)


if __name__ == "__main__":
    time_size = 5
    batch_size = 100
    input_size = 13  # 一般是词向量维度[特征维度]
    output_size = 60  # 一般是 vocab_size
    hidden_size = 8
    x = Variable(np.random.randn(batch_size, time_size, input_size))
    time_rnn = TimeRNN(hidden_size=hidden_size)
    y = time_rnn(x)
    print("TimeRNN输出形状:", y.shape)
    y.backward()
    print(x.grad.shape)

    x = Variable(np.random.randn(batch_size, time_size, input_size))
    simple_time_rnn = TimeSimpleRNN(hidden_size=hidden_size, output_size=output_size)
    y = simple_time_rnn(x)
    print("SimpleTimeRNN输出形状:", y.shape)
    y.backward()
    print(x.grad.shape)

    t = Variable(np.random.randint(0, output_size, size=(batch_size, time_size)))
    loss = time_softmax_cross_entropy(y, t)
    print("TimeSoftmaxCrossEntropy损失:", loss.value)
    loss.backward()
    print(y.grad.shape)
