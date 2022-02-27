# 线性回归简洁实现

import numpy as np
import torch
from torch.utils import data
from  d2l import torch as d2l
# nn是神经网络
from torch import nn

#生成数据集
true_w = torch.tensor([2,-3.4])
true_b = 4.2
feature,labels = d2l.synthetic_data(true_w,true_b,1000)

# 构造一个pytorch数据迭代器
def load_array(data_array,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_array)
    # dataset拿到数据集，DataLoader可以跟据dataset获取batch_size个数据，shuffle决定是否打乱顺序
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((feature,labels),batch_size)

# 转成python的iterater
next(iter(data_iter))

# 使用框架定义好的层
# 输入维度为2，输出为1
# 放到sequential的层，相当于layer的list
net = nn.Sequential(nn.Linear(2,1))

# 初始化模型参数
# linear层在第1层 net[0]相当于Linear
# normal_ 是使用正态分布来替换值 均值为0，方差为0.01
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

# 使用均方误差类 MSELoss 作为损失函数
loss = nn.MSELoss()

# 优化算法 随机梯度下降 实例化SGD类 传入所有参数和学习率
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

# 训练
num_epochs = 10
for epoch in range(num_epochs):
    for x,y in data_iter:
        l = loss(net(x),y)
        trainer.zero_grad()
        # l已经计算了sum
        l.backward()
        # 调用step进行模型更新
        trainer.step()
    # 把所有的fetures和labels放进loss里面，打印
    l = loss(net(feature),labels)
    print(f'epoch {epoch+1} , loss {l:f}')



