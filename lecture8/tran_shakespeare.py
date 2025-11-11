import time

import numpy as np

from lecture8.core import Dataset, accuracy, Adam, SeqDataLoader, softmax_cross_entropy
from lecture8.lstm import LSTMWithEmbedding


class TinyShakespeareDataset(Dataset):

    def prepare(self):
        # 路径可改
        file_path = 'tiny_shakespeare.txt'
        with open(file_path, 'r') as f:
            data = f.read()
        chars = list(data)

        char_to_id = {}
        id_to_char = {}
        for word in data:
            if word not in char_to_id:
                new_id = len(char_to_id)
                char_to_id[word] = new_id
                id_to_char[new_id] = word

        indices = np.array([char_to_id[c] for c in chars])
        self.data = indices[:-1]
        self.label = indices[1:]
        self.char_to_id = char_to_id
        self.id_to_char = id_to_char


if __name__ == "__main__":
    # 设置一些超参数
    max_epoch = 10
    batch_size = 10
    hidden_size = 100
    bptt_length = 30
    embedding_size = 32

    train_set = TinyShakespeareDataset(train=True)
    test_set = TinyShakespeareDataset(train=False)

    train_loader = SeqDataLoader(train_set, batch_size)
    test_loader = SeqDataLoader(test_set, batch_size)

    output_size = len(train_set.char_to_id)  # 一般是 vocab_size
    seqlen = len(train_set)

    #   输出采用数字编码，在计算损失时会将其转换为 one-hot 编码
    model = LSTMWithEmbedding(hidden_size, output_size, embedding_size)
    optimizer = Adam(model)

    # 检查一批样本
    for x, t in train_loader:
        print("一个rnn块的输入序列（整数编码）:", x.shape)
        print("一个rnn块的目标序列（整数编码）:", t.shape)
        break

    for epoch in range(max_epoch):
        # ---- 训练阶段 ----
        s1 = int(round(time.time()))
        model.reset_state()  # 先重置一下隐含状态
        total_loss, total_acc = 0, 0
        count = 0

        for x, t in train_loader:
            # x和t 的维度本应该是 (batch_size, time_size, input_dim)
            # 但这里字符级编码，所以 input_dim 为 1 省略， 所以这里 x为 (batch_size, time_size)
            y = model(x)
            # 输出 y 的维度应该是 (batch_size, time_size, 种类数 vocab_size )
            total_loss += softmax_cross_entropy(y, t)
            acc = accuracy(y, t)  # 新增：计算当前批次的准确率
            total_acc += float(acc.value) * len(t)  # 新增：累加准确率

            count += 1

            if count % bptt_length == 0 or count == seqlen:
                model.clear_grads()
                total_loss.backward()
                total_loss.unchain_backward()
                optimizer.update()
        avg_loss = float(total_loss.value) / count
        avg_acc = total_acc / (count * batch_size)  # 新增：计算平均准确率
        time_end = time.time()
        print('| epoch %d | loss %f | acc %f | time %f' % (epoch + 1, avg_loss, avg_acc, time_end - s1))

    # # ---- 测试阶段 ----
    # total_loss, total_acc = 0, 0
    # count = 0
    # for x, t in test_loader:
    #     y = model(x)
    #     loss = time_softmax_cross_entropy(y, t)
