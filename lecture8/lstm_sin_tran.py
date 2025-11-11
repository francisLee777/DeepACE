import matplotlib.pyplot as plt
import numpy as np

from lecture8.core import Dataset, mean_squared_error, Adam
from lecture8.lstm import SimpleLSTM

# 数值溢出主动报错
np.seterr(over='raise')


class SinCurve(Dataset):
    def prepare(self):
        num_data = 1000
        dtype = np.float32
        # 生成 1000个 0 到 2π 之间的等间隔数据点
        x = np.linspace(0, 2 * np.pi, num_data)
        # 添加噪声值
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise  # 加入噪声
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]  # (1000,1)
        self.label = y[1:][:, np.newaxis]  # (1000,1)


# Hyperparameters 超参数，需要人工手动设置
max_epoch = 20  # 训练轮数。每一轮是一次完整的数据集遍历
hidden_size = 100
bptt_length = 30  # 截断反向传播的时间步长。每 bptt_length 个时间步进行一次反向传播更新

train_set = SinCurve(train=True)
seqlen = len(train_set)  # 每次从数据集中取样本的序列长度，这里直接取整个数据集的长度

# 预测数据值，回归任务，所以 output_size 是 1
model = SimpleLSTM(hidden_size, 1)
optimizer = Adam(model)  # 使用 Adam 效果更好

# Start training.
for epoch in range(max_epoch):
    # 每轮训练需要重置 RNN 层的隐藏状态
    model.reset_state()
    loss, count = np.float32(0), 0

    for x, t in train_set:
        # x 的形状是 (1, )，需要 reshape 成 (1, 1). 因为 RNN 的入参需要是 2 维的，否则无法做 linear 中的矩阵乘法
        x = x.reshape(1, 1)
        y = model(x)
        loss += mean_squared_error(y, t)  # 数值类型的回归任务，使用均方误差损失函数即可。
        count += 1

        # 当遍历一个 bptt_length 长度的序列时，需要切断 RNN 层的状态
        if count % bptt_length == 0 or count == seqlen:
            model.clear_grads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.value) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

# Plot
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()  # 重置模型，消除训练时的h状态
pred_list = []

for x in xs:
    x = np.array(x).reshape(1, 1)
    y = model(x)
    pred_list.append(y.value[0][0])

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
