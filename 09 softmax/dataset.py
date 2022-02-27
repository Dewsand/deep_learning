# 使用Fashion-MNIST数据集 图像分类数据集

import torch
import torchvision
from  torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 通过框架的内置函数将Fashion-MINIST数据集下载并读取到内存中
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
# 把dataset下载，下载训练数据，下载得到Tensor类型，从网上下载
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=False)
# 测试数据集，验证模型好坏数据集，False代表下载测试数据集
mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=False)

print("len",len(mnist_train),",",len(mnist_test))
print("shape",mnist_train[0][0].shape,",",mnist_test[0][0].shape)

# Fashion-MNIST中包含的10个类别分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。以下函数用于在数字标签索引及其文本名称之间进行转换。

# 返回Fashion-MNIST数据集的文本标签
def get_fashion_mnist_labels(labels):
    test_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]

# 显示图像列表
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize = (num_cols*scale,num_rows*scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

# 显示图片
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
d2l.plt.show()

print("x.shape",X.shape,",y.shape",y.shape)

# 读取一小批量，大小为batch_size
batch_size = 256

# 使用4个进程来读取的数据
def get_dataloader_workers():
    return 4

train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


# 整合以上所有 resize是否需要把图片变得更大
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))