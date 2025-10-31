import numpy as np

from lecture7.core import Linear, Layer, sigmoid

if __name__ == "__main__":
    # 假设有两层神经网络参数
    W1 = np.random.randn(100, 50).astype(np.float32)  # 输入层到隐藏层
    b1 = np.random.randn(50).astype(np.float32)
    W2 = np.random.randn(50, 10).astype(np.float32)  # 隐藏层到输出层
    b2 = np.random.randn(10).astype(np.float32)

    # ---------- 保存 ----------
    np.savez_compressed("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    print("✅ 权重参数已压缩保存到 weights.npz")

    # ---------- 加载 ----------
    data = np.load("weights.npz")

    print("W1 形状:", data["W1"].shape)  # (100, 50)
    print("b2 示例:", data["b2"][:5])  # [ 0.33 -1.12  0.95  0.24 -0.77]


    class TwoLayerNet(Layer):
        def __init__(self, hidden_size, output_size, dtype=np.float32):
            super().__init__()
            self.l1 = Linear(hidden_size, dtype=dtype)
            self.l2 = Linear(output_size, dtype=dtype)

        def forward(self, x):
            return self.l2(sigmoid(self.l1(x)))


    t = TwoLayerNet(3, 2)
    # t.forward(np.random.randn(100, 3))  # 前向传播一次，确保参数被初始化
    # t.save_params()
    t.load_params()
    t.forward(np.random.randn(100, 3))  # 前向传播一次，确保参数被初始化
