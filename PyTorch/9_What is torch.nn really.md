本节将对对各种神经网络层、[torch.nn](https://pytorch.org/docs/stable/nn.html)类、[torch.optim](https://pytorch.org/docs/stable/optim.html)类、 [Dataset](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset)类、[DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)类再次讲解——showing exactly what each piece does, and how it works to make the code either more concise, or more flexible.

# Neural net from scratch (no torch.nn)

我们使用 [MNIST](http://deeplearning.net/data/mnist/) 数据集（黑白手写0-9数字图片）。

```python
# 1 属性命名
from pathlib import Path  # 使用pathlib处理路径
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
```



```python
# 2 数据集下载
import pickle  # pickle是一种用于序列化数据的Python特定格式
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    # 此数据集为numpy数组格式，并已使用pickle存储
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
```



```python
# 3 处理图像
from matplotlib import pyplot
import numpy as np

# 每个图像的像素为28*28，被存储为长度为784(=28x28)的扁平行；所以我们需要将其重塑为2D图像
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# print(x_train.shape)
```



```python
# 4 把numpy数组格式的数据集转换成Tensor类型
import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
```



```python
# 5 手动设置权重和偏置
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()  # weights.requires_grad为True
bias = torch.zeros(10, requires_grad=True)
```



```python
# 6
# 6.1 定义一个激活函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)
# 6.2 定义一个简单的线性模型（使用了普通矩阵乘法和广播加法）
def model(xb):
    return log_softmax(xb @ weights + bias)  # @代表点积
```



```python
# 7 准备一小批数据进行一次预测
bs = 64       	   # batch size
xb = x_train[0:bs]  
preds = model(xb)  # predictions(预测结果)
print(preds[0], preds.shape)
```

> 输出结果：`tensor([-2.3410, -2.0094, -2.1276, -2.0898, -2.4220, -2.4902, -3.1141, -3.0836, -1.7615, -2.3727], grad_fn=<SelectBackward>) torch.Size([64, 10])`. 可以看到`preds[0]`的输出不仅包括预测出的10个值，还包含了一个梯度函数`grad_fn`，反向传播时会用到此函数。

```python
# 8 手动实现负对数似然作为损失函数
def nll(input, target):
  return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))
```



```python
# 9 手动实现计算模型精确度的函数
def accuracy(out, yb):
  preds = torch.argmax(out, dim=1)
  return (preds == yb).float().mean()

print(accuracy(preds, yb))
```

> 对比8、9产生了疑问：损失值和精确度都可以用来衡量模型的好坏吧，所以只保留其中一个不就可以了吗？

```python
# 10 循环进行训练
from IPython.core.debugger import set_trace  # 开启Python的debugger模式，以便看到每一步值的变化

lr = 0.5
epochs = 2
for epoch in range(epochs):
  for i in range((n-1) // bs + 1):
    # set_trace()
    start_i = i * bs
    end_i = start_i + bs
    xb = x_train[start_i:end_i]
    yb = y_train[start_i:end_i]
    pred = model(xb)
    loss = loss_func(pred, yb)

    loss.backward()
    with torch.no_grad():
      weights -= weights.grad * lr
      bias -= bias.grad * lr
      weights.grad.zero_()
      bias.grad.zero_()


# 经过这一循环训练，损失值就降得很小了
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
```

# 使用torch.nn优化

下面将使用Pytorch的`nn`模块对上面的程序进行修改(1、2、3、4步无需改动)，从而使得程序更短、更容易理解、更灵活。

## nn.functional

`torch.nn.functional`模块中不仅含有各种损失和激活函数，还有用于创建神经网络的便捷函数，例如池化函数（也有用于进行卷积、线性层等的函数，==但正如我们将看到的，使用库的其他部分通常可以更好地处理这些==）。 

在<font color=red>第6步</font>中，我们手动定义了激活函数和损失函数。现在我们使用`torch.nn.functional`模块中已经定义好了的函数。

```python
import torch.nn.functional as F

loss_func = F.cross_entropy  # F.cross_entropy函数将负对数似然损失和对数softmax激活结合起来

def model(xb):
  return xb @ weights + bias
```

## nn.Module

` nn.Module `是一个能够跟踪状态的类，该类中有许多我们将使用的属性和方法（例如`.parameters()`和`.zero_grad()`）。

在<font color=red>第5步</font>中，我们手动设置了权重和偏置。现在我们创建一个` nn.Module `的子类来保存我们的权重、偏差和前进步骤的方法。

```python
from torch import nn

class Mnist_Logistic(nn.Module):
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
    self.bias = nn.Parameter(torch.zeros(10))

  def forward(self, xb):
    return xb @ self.weights + self.bias

model = Mnist_Logistic()  # 实例化一个神经网络模型
```

使用`.parameters()`和`.zero_grad()`来替代<font color=red>第10步</font>中参数更新的语句：

```python
bs = 64       	   		 # batch size
xb = x_train[0:bs]  
print(loss_func(model(xb), yb))  # 模型训练前

def fit():
  for epoch in range(epochs):
    for i in range((n-1) // bs + 1):
      start_i = i * bs
      end_i = start_i + bs
      xb = x_train[start_i:end_i]
      yb = y_train[start_i:end_i]
      pred = model(xb)
      loss = loss_func(pred, yb)

      loss.backward()
      with torch.no_grad():  			
        for p in model.parameters():  # .parameters()
          p -= p.grad * lr
        model.zero_grad()			  # .zero_grad()

fit()
print(loss_func(model(xb), yb))  # 模型训练后（损失值降低）
```

> 使代码更简洁，更不容易出现忘记更新某些参数的错误。

## nn.Linear

在上一节使用`nn.module`优化后，我们还可以进一步优化代码：不手动定义和初始化`self.weights`、`self.bias`，也不在正向传播函数中手动计算`xb @ self.weights + self.bias`，而是使用`nn.Linear`定义一个线性层，其内部已经为我们完成了那些工作。

> Pytorch has many types of predefined layers like nn.Linear that can greatly simplify our code, and often makes it faster too.

再次优化<font color=red>第5步</font>：

```python
class Mnist_Logistic(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(784, 10)
  
  def forward(self, xb):
    return self.lin(xb)

model = Mnist_Logistic()
```

## torch.optim

`torch.optim`中有许多优化算法，在上上一节使用`nn.module`优化后，我们还可以进一步优化代码：不手动更新参数，而直接使用优化类的`step()`和`zero_grad()`。

将实例化的神经网络模型和优化器封装在一起：

```python
from torch import optim
def get_model():  # 返回一个神经网络的实例模型和一个优化器
  model = Mnist_Logistic()
  return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
```

再次优化<font color=red>第10步</font>——使用优化器的`step()`和`zero_grad()`：

```python
bs = 64       	  # batch size
xb = x_train[0:bs]  
print(loss_func(model(xb), yb))  # 模型训练前

for epoch in range(epochs):
  for i in range((n-1) // bs + 1):
    start_i = i * bs
    end_i = start_i + bs
    xb = x_train[start_i:end_i]
    yb = y_train[start_i:end_i]
    pred = model(xb)
    loss = loss_func(pred, yb)

    loss.backward()
    opt.step()		 # 使用优化器的step()
    opt.zero_grad()  # 使用优化器的zero_grad()
    
print(loss_func(model(xb), yb))  # 模型训练后
```

## Dataset

PyTorch有一个抽象的Dataset类。Dataset类可以是任何具有`__len__`函数（由Python的标准`len`函数调用）和`__getitem__`函数作为索引方式的任何东西。

PyTorch 的TensorDataset是一个包装了tensors的Dataset。通过定义长度和索引方式，为我们提供了一种沿张量的第一维迭代、索引和切片的方法。这将使我们在训练的同一行中更容易访问自变量和因变量。

再再次优化<font color=red>第10步</font>：

```python
from torch.utils.data import TensorDataset      # 新语句

train_ds = TensorDataset(x_train, y_train)      # 新语句

bs = 64       	  # batch size
xb = x_train[0:bs]  
print(loss_func(model(xb), yb))  # 模型训练前

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]  # 新语句
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
        
print(loss_func(model(xb), yb))  # 模型训练后
```

## DataLoader

Pytorch的`DataLoader`负责管理批处理。您可以从任何一个`Dataset`创建一个`DataLoader`。 `DataLoader`可以更轻松地迭代批次。`DataLoader`不必使用`train_ds[i*bs : i*bs+bs]`，而是自动为我们提供每个小批量。

再再再次优化<font color=red>第10步</font>：

```python
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader   		# 新语句

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)  # 新语句

bs = 64       	  # batch size
xb = x_train[0:bs]  
print(loss_func(model(xb), yb))  # 模型训练前

for epoch in range(epochs):
    for xb, yb in train_dl:   		      		# 新语句
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
        
print(loss_func(model(xb), yb))  # 模型训练后
```

## 验证集

- 验证集(a validation set): 用来确定是否过度拟合。

- 混洗(shuffling)训练数据对于防止批次和过度拟合之间的相关性很重要（不需混洗验证数据）。

- 我们将使用比训练集大两倍的验证集批量大小。这是因为验证集不需要反向传播，因此占用的内存更少（不需要存储梯度）。我们利用这一点来使用更大的批量大小并更快地计算损失。
- 注意，在训练之前调用`model.train()`，在推理之前调用`model.eval()`，因为它们被`nn.BatchNorm2d`和`nn.Dropout`等层使用，以确保这些不同阶段的适当行为。

```python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
    	pred = model(xb)
    	loss = loss_func(pred, yb)
        
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
      	valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dll)

    print(epoch, valid_loss / len(valid_dl)) # 打印验证集的损失值
```

## 函数封装

对代码进行封装，达到“高内聚、低耦合”的效果。

1. `get_data()`：返回训练集和验证集的数据加载器

```python
def get_data(train_ds, valid_ds, bs):
    return (
    	DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2)
    )
```

2. `loss_batch()`：计算一批次数据的损失值

```python
'''
参数opt：
	对于训练集，需要传入优化器，并使用它来执行反向传播。
	对于验证集，不需传入优化器，因为不需要执行反向传播。
'''
def loss_batch(def (, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
               
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
               
    return loss.item(), len(xb)
```

3. `fit()`：训练模型并计算每个时期的训练和验证损失

```python
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:  # 训练模型
            loss_batch(model, loss_func, xb, yb, opt)
            
    model.eval()
    with torch.no_grad():    # 计算验证集的损失值
        losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(sums)		# ？？？
    print(epoch, val_loss)
```

- 运行上面的函数：

```python
train_dl, valid_dl = get_data(train_ds, valid_ds, bss)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

事实上，您可以使用这 3 行基本代码来训练各种模型。让我们看看我们是否可以使用它们来训练卷积神经网络 (CNN，convolutional neural network)！

## Switch to CNN

用三个卷积层构建我们的神经网络：

```python
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # 卷积层
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        
        def forward(self, xb):
            xb = xb.view(-1, 1, 28, 28)
            xb = F.relu(self.conv1(xb))  #  Each convolution is followed by a ReLU. 
            xb = F.relu(self.conv2(xb))
            xb = F.relu(self.conv3(xb))
            xb = F.avg_pool2d(xb, 4)  # At the end, we perform an average pooling.
            return xb.view(-1, xb.size(1))
        
lr = 0.1
```

动量（[Momentum](https://cs231n.github.io/neural-networks-3/#sgd)）是随机梯度下降的一种变体，它也考虑了先前的更新，通常会获得更快的训练。

```python
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## nn.Sequential

`Sequential`对象以顺序方式运行其中包含的每个模块。这是编写神经网络的一种更简单的方法。

现在，我们自定义一个视图层`Lambda`，然后在使用`Sequential`定义网络时使用该层：

```python
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)
```

用`Sequential`定义神经网络：

```python
model = nn.Sequential(
	Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1))
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## Wrapping  DataLoader

==不太理解==

我们的 CNN 相当简洁，但它只适用于 MNIST，因为： 

- 它假设输入是一个 28 X 28 长的向量；

- 它假设最终的 CNN 网格大小是 4*4（因为这是我们使用的平均池化内核大小）。

让我们摆脱这两个假设，从而使我们的模型适用于任何二维单通道图像。首先，我们可以通过将数据预处理移动到生成器中来移除初始 Lambda 层：

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
    
    def __len__(self):
		return len(self.dl)
    
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield(self.func(*b))
            
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

接下来，我们可以用 `nn.AdaptiveAvgPool2d` 替换 `nn.AvgPool2d`，这允许我们定义我们想要的输出张量的大小，而不是我们拥有的输入张量。因此，我们的模型将适用于任何尺寸的输入。

```python
model = nn.Sequential(
	nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1))
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

训练：

```python
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

## Using your GPU

If you’re lucky enough to have access to a CUDA-capable GPU (you can rent one for about $0.50/hour from most cloud providers) you can use it to speed up your code. First check that your GPU is working in Pytorch:

```python
print(torch.cuda.is_available())
```

输出结果为True或False。

如果输出结果为True，就可以创建一个设备对象了：

```python
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
```

让我们更新`preprocess`以将批次移动到 GPU：

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

Finally, we can move our model to the GPU.

```python
model.to(dev)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

You should find it runs faster now:

```python
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
```

# 总结

- torch.nn
  - `Module`: creates a callable which behaves like a function, but can also contain state(such as neural net layer weights). It knows what `Parameter` (s) it contains and can zero all their gradients, loop through them for weight updates, etc.
  - `Parameter`: a wrapper for a tensor that tells a `Module` that it has weights that need updating during backprop. Only tensors with the requires_grad attribute set are updated
  - `functional`: a module(usually imported into the `F` namespace by convention) which contains activation functions, loss functions, etc, as well as non-stateful versions of layers such as convolutional and linear layers.
- `torch.optim`: Contains optimizers such as `SGD`, which update the weights of `Parameter` during the backward step
- `Dataset`: An abstract interface of objects with a `__len__` and a `__getitem__`, including classes provided with Pytorch such as `TensorDataset`
- `DataLoader`: Takes any `Dataset` and creates an iterator which returns batches of data.