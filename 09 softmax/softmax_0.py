# softmax从0开始实现

import torch
from IPython import display
from d2l import torch as d2l

# 读取数据集，批量大小为256
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

# 之前每张图片长宽都为28，通道为1，需要将图片展平成向量，28*28*1=784
# 数据集有10类，则输出维度为10
num_inputs = 784
num_outputs = 10

# 权重w初始化为高斯随机数，均值0，方差1
w = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
# 偏移 长为10的向量 需要计算梯度
b = torch.zeros(num_outputs,requires_grad=True)

# 实现softmax 矩阵对每一行进行softmax
# softmax(X) ij=exp(X ij)/sum k(exp(Xik))
def softmax(X):
    X_exp = torch.exp(X)
    # 按每一行求和，得到列向量
    partition = X_exp.sum(1,keepdim=True)
    # 使用广播机制进行除法
    return X_exp / partition

# 验证softmax
# X = torch.normal(0, 1, (2, 5))
# print(X)
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))

# 定义模型
def net(X):
    # 输入需要一个批量大小*输入维数的矩阵，-1是批量大小，X被reshape成256*784的矩阵，再对x对权重进行矩阵乘法，然后通过广播机制加上偏移
    return softmax(torch.matmul(X.reshape((-1,w.shape[0])),w)+b)

# 交叉熵作损失函数

# 创建数据y_hat,包含2个样本在3个类别的预测概率，用y作为y_hat中概率的索引
y = torch.tensor([0,2]) #两个样本概率的索引为0和2
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]]) #模拟softmax的输出，为样本在所有类别的概率
print(y_hat[[0,1],y])   #跟据索引找到概率输出

# 实现交叉熵损失函数
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

print(cross_entropy(y_hat,y))

# 将预测类别与真实y元素进行比较
# 计算正确的数量
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat,y) / len(y))

class Accumulator:  #@save
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 计算在指定数据集上模型的精度
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()  #将模型设置为评估模式
    metric = Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0] / metric[1]


# print(evaluate_accuracy(net,test_iter))

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）。"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):    #如果是nn模具实现
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)  #交叉熵损失函数
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:   #从0开始实现
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练准确率 所有loss累加/样本总数，正确分类正确的、样本总数
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  #@save
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)




def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1

def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
d2l.plt.show()



