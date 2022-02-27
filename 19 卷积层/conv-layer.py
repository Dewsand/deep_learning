import torch
from torch import nn
from d2l import torch as d2l

# 计算二维互相关运算 X是输入， K是核矩阵
def corr2d(X, K):
    h,w = K.shape #行数列数
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # X的小区域矩阵和核矩阵进行乘法求和
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 验证二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 卷积层的一个简单应用： 检测图像中不同颜色的边缘
X = torch.ones((6,8))
X[:, 2:6] = 0
print("\n",X)

K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X,K)
print("\n",Y)

# 卷积核 K 只能检测垂直边缘
print("\n",corr2d(X.t(), K))


# 学习由X生成Y的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1,2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    # 梯度下降
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')

print("\n",conv2d.weight.data.reshape((1,2)))

