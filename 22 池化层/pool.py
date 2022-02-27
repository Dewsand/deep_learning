import torch
from torch import nn
from d2l import torch as d2l

# 实现池化层的正向传播
def pool2d(X, pool_size, mode='max'):
    p_h,p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h +1, X.shape[1] - p_w +1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

# 验证池化层
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))

print("\n",pool2d(X, (2, 2), 'avg'))

# 填充和步幅
X = torch.arange(16, dtype=float).reshape(((1,1,4,4)))
print("\n",X)

# 深度学习框架中的步幅与池化窗口的大小相同  3x3
pool2d = nn.MaxPool2d(3)
print("\n",pool2d(X))

# 可以指定填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("\n",pool2d(X))

# 任意大小的池化窗口
pool2d = nn.MaxPool2d((2,3), padding=(1,1), stride=(2,3))
print("\n",pool2d(X))

# 池化层在每个输入通道上单独运算
X = torch.concat((X, X+1),1)
print("\n",X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print("\n",pool2d(X))