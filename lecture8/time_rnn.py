import numpy as np

from lecture8.core import Linear, tanh, Layer, stack, Variable, softmax, logsumexp, Function, Model, DataLoader


# 暂时废弃

class TimeRNN(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.x_w = Linear(hidden_size)
        self.h_prev_w = Linear(hidden_size, need_bias=False)
        self.h_cur = None

    def reset_state(self, h=None):
        self.h_cur = h

    def forward(self, x):
        # self.reset_state()

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


class TimeSimpleRNN(Model):
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
        x, t = self.input_variable
        N, T, C = x.shape
        dy *= np.float32(1 / (N * T))

        # softmax 前向
        y = softmax(x.reshape(N * T, C))  # y 仍然是 Variable
        y = y.value  # .copy()  # 拿出数值副本，打断计算图，安全操作

        # 修改标签对应项，避免构造 one-hot
        t_flat = t.value.reshape(N * T)
        y[np.arange(N * T), t_flat] -= 1

        gx = y * dy  # y 是 np.array, dy 是 Variable
        gx = gx.reshape(N, T, C)
        return gx, None


def time_softmax_cross_entropy(x, t):
    return TimeSoftmaxCrossEntropy()(x, t)


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t


class TimeDataLoader:
    def __init__(self, dataset, batch_size, seq_len):
        """
        dataset: 时序数据集，例如 TinyShakespeareDataset
        batch_size: 并行序列数（多少条时间序列并行训练）
        seq_len: 一波次数据输入数量，也是每次反向传播的时间块长度（truncated BPTT）
        """
        self.dataset = dataset
        self.data = dataset.data
        self.seq_len = seq_len
        self.batch_size = batch_size

        # 确保数据长度能均分成 batch_size 条序列
        self.data_size = len(self.data)
        self.jump = self.data_size // batch_size  # 每个序列的起始偏移
        self.offsets = [i * self.jump for i in range(batch_size)]  # 每个序列的初始起点

        self.time_idx = 0  # 当前时间位置
        self.max_time = self.jump - seq_len - 1  # 最后一次能取到的位置

    def __iter__(self):
        return self

    def __next__(self):
        if self.time_idx >= self.max_time:
            self.time_idx = 0
            raise StopIteration

        x_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)
        t_batch = np.zeros((self.batch_size, self.seq_len), dtype=np.int32)

        for i, offset in enumerate(self.offsets):
            start = offset + self.time_idx
            end = start + self.seq_len + 1
            chunk = self.data[start:end]
            x_batch[i] = chunk[:-1]
            t_batch[i] = chunk[1:]

        self.time_idx += self.seq_len
        return x_batch, t_batch

    def reset(self):
        self.time_idx = 0


if __name__ == "__main__":
    time_size = 5
    batch_size = 100
    input_size = 13  # 一般是词向量维度[特征维度]
    output_size = 60  # 一般是 vocab_size
    hidden_size = 8
    x = Variable(np.random.randn(batch_size, time_size, input_size).astype(np.float32))
    time_rnn = TimeRNN(hidden_size=hidden_size)
    y = time_rnn(x)
    print("TimeRNN输出形状:", y.shape)
    y.backward()
    print(x.grad.shape)

    x = Variable(np.random.randn(batch_size, time_size, input_size).astype(np.float32))
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
