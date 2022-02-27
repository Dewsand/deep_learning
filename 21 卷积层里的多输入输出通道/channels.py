import torch
from torch import nn
from d2l import torch as d2l

# 多输入通道互相关运算
def corr2d_multi_in(X, K):
    # 计算对应通道的互相关运算，求和
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

# 计算多个通道的输出的互相关函数
def corr2d_multi_in_out(X,K):
    # 对每个输入X，与每一层k做互相关运算，在0的维度基础上堆起来
    return torch.stack([corr2d_multi_in(X, k) for k in K],0)

K = torch.stack((K, K+1, K+2),0)
print("\n",K.shape)

print("\n",corr2d_multi_in_out(X,K))


# 1x1卷积等价于一个全连接
def corr2d_multi_in_out_1x1(X,K):
    c_i,h,w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6