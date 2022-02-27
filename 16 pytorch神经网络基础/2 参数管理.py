import torch
from torch import nn

# 具有单隐藏层的多层感知机
# 在net中，net[0]是nn.Linear(4,8),net[1]是nn.ReLU(), net[2]是nn.Linear(8,1)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))

X = torch.rand(size=(2,4))
print(net(X))

# 参数访问 state_dict 是模型参数 权重8x1 偏差是常量
print("\nstate_dict")
print(net[2].state_dict())

# 目标参数
print("\nbias")
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 还没有做反向计算
print("\nweight")
print(net[2].weight.grad == None)

# 一次性访问所有参数
print("\nall parameter")
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])

# 通过名字访问特定参数
print("\n2.bias")
print(net.state_dict()['2.bias'].data)

# 从嵌套快中手机参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1))
print("\nrgnet")
print(rgnet)
print(rgnet(X))

# 内置初始化 m是nn.Module
def init_normal(m):
    if type(m) == nn.Linear:
        # 是全连接层，对权重weight作均值为0，方差为0.01的初始化
        # init.normal_是替换操作,作初始化操作
        nn.init.normal_(m.weight,mean=0,std=0.01)
        # 偏差bias赋值为0
        nn.init.zeros_(m.bias)

# apply 是对net里面所有的layer，遍历一遍，嵌套的也嵌套遍历
# net = rgnet
net.apply(init_normal)
print("\ninitnormal")
print(net[0].weight.data[0],net[0].bias.data[0])
print("\nall parameters")
print(*[(name,param.data) for name,param in net.named_parameters()])
# print(net.state_dict())

# 初始化weight为1
def init_constant(m):
    if type(m) == nn.Linear:
        # 是全连接层，对权重weight赋值为1
        nn.init.constant_(m.weight,1)
        # 偏差bias赋值为0
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print("\ninitconstant")
print(net[0].weight.data[0],net[0].bias.data[0])


# 对某些块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

print("\ndifferent initial")
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in net[0].named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        # 绝对值 >= 5 的保留，否则设置为0
        m.weight.data *= m.weight.data.abs() >= 5

print("\nmy_init")
net.apply(my_init)
print(net[0].weight[:2])


# 直接设置
print()
net[0].weight.data[:] += 1
net[0].weight.data[0,0] = 42
print(net[0].weight.data[0])


# 参数绑定 -- 在不同网络共享参数的方法
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))
net(X)
print()
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])