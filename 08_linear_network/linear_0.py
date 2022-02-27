# 线性回归从0开始实现

import random
import torch
from d2l import torch as d2l

# 构造数据集
def synthetic_data(w,b,num_examples):
    print("------构造数据集---------")
    # 生成 y=Xw+b+噪声
    # x是一个均值为0，方差为1的随机数，行数为样本数，列数为w的长度
    x = torch.normal(0,1,(num_examples,len(w)))

    print("构造数据集的x.shape",x.shape)

    y = torch.matmul(x,w)+b

    print("构造数据集的y.shape",y.shape)

    y += torch.normal(0,0.01,y.shape)

    # print("添加噪声之后的y",y)

    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2

fetures,labels = synthetic_data(true_w,true_b,1000)

print("fetures",fetures.shape)
print("labels",labels.shape)

# d2l.set_figsize()
# d2l.plt.scatter(fetures[:,1].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()

# 每次读取一个小批量
def data_iter(bach_size,fetures,labels):
    num_examples = len(fetures)
    indices = list(range(num_examples))
    # 样本随机抽取，没有特定的顺序
    random.shuffle(indices)
    for i in range(0,num_examples,bach_size):
        batch_indices = torch.tensor(indices[i:min(i+bach_size,num_examples)])
        yield fetures[batch_indices],labels[batch_indices]

batch_size = 10

# 每次获取10个特征和标签
for x,y in data_iter(batch_size,fetures,labels):
    print(x,"\n",y)
    break

# 定义初始化模型参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# 定义模型
def linreg(x,w,b):
    # 线性回归模型
    return torch.matmul(x,w)+b


# 定义损失函数（均方误差）
def squares_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 定义优化算法 参数（list，w,b）,学习率，批量大小
def sgd(params,lr,batch_size):
    # 小批量随机梯度下降
    # 不需要计算梯度，更新不需要梯度计算
    with torch.no_grad():
        for param in params:
            # 参数减去学习率乘梯度求平均
            param -= lr * param.grad / batch_size
            # 梯度清0，pytorch不会自动清除梯度
            param.grad.zero_()


# 训练
lr = 0.03
num_epochs = 10
net = linreg
loss = squares_loss

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,fetures,labels):
        l = loss(net(x,w,b),y)  #求x和y的小批量损失
        # 因为 l 的形状是[batch_size,1],而不是一个标量，求和然后算[w,b]梯度
        l.sum().backward()
        sgd([w,b],lr,batch_size) #使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(fetures,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 通过对比真实的w，b进行比较
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')