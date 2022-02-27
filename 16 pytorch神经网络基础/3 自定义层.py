import torch
import torch.nn.functional as F
from torch import nn

# 构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # 输入减去均值，使均值变成0
        return X - X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

# 将层作为组件合并到构建更复杂的模型中
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())

Y = net(torch.rand(4,8))
print("\n",Y.mean())

# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # randn是正态分布
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5,3)
print("\n",dense.weight)

# 使用自定义层直接执行正向传播计算
print("\n",dense(torch.rand(2,5)))

# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print("\n",net(torch.rand(2,64)))