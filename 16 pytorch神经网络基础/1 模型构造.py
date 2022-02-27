import torch
from torch import nn
from torch.nn import functional as F

# 构建一个简单的神经网络 线性层+ReLU+线性层
# nn.Sequential定义了一种特殊的 Module
# 任何一个层或者神经网络都是 Module 的子类
net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

X = torch.rand(2,20)
print(net(X))

# 自定义快
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  #初始化
        # 定义两个全连接层 隐藏层和输出层
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
print(net(X))

# 定义顺序快
class Mysequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net = Mysequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(net(X))

# 在正向传播函数种执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight)+1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
        # return X

net = FixedHiddenMLP()
print(net(X))

# 混合搭配各种组合快的方法 嵌套使用
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self, X):
        return self.linear(self.net(X))

chimra = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print(chimra(X))