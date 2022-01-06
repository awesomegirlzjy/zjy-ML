# torch.autograd
> 学习地址：https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

`torch.autograd`是PyTorch的**自动差分引擎（ automatic differentiation engine ）**，可为神经网络训练提供支持。在本节中，您将获得有关【autograd如何帮助神经网络训练】的概念性理解。

神经网络（NN）是在某些输入数据上执行的**嵌套函数**的集合。这些函数由参数（由**权重**和**偏差**组成）定义，这些参数存储在张量中。

训练神经网络分为两个步骤：
1. 正向传播(Forward Propagation)：做出预测结果。
2. 反向传播(Backward Propagation)：调整参数——从输出向后遍历，收集关于函数参数的误差**导数**（梯度）并使用**梯度下降**优化参数来实现此目的。
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210507175059921.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
>
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210507175122285.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)

流程代码示例：
```python
import torch, torchvision


'''
1、从torchvision包中加载预训练的resnet18模型;
   创建一个有三个通道、64*64大小的随机张量;
   相应的标签也为随机值。
'''
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)


# 2、通过模型的每个层运行输入数据以进行预测(forward pass)
prediction = model(data)


'''
3、计算损失loss；
   对损失进行后向传播——loss调用backward()；
   Autograd计算每个模型参数的梯度并将其存储在参数的.grad属性中。
'''
loss = (prediction - labels).sum()
loss.backward()


# 4、加载优化器
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


# 5、调用step()启动梯度下降，优化器会通过存储在.grad属性中的梯度来调整每个参数
optim.step()
```
## Differentiation in Autograd
```python
import torch
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)  # 由于Q是个向量，所以我们需要显式的传一个梯度参数

print(a.grad)  # 对a的偏导数
print(b.grad)  # 对b的偏导数
```
# Autograd
训练神经网络时，最常用到的算法是**反向传播（ back propagation）**。在该算法中，根据**损失函数（loss function）**相对于给定参数的**梯度（gradient ）** 来调整参数（模型权重）。

PyTorch具有一个内置的差异化引擎，称为`torch.autograd`，它可以*自动计算梯度*。

考虑最简单的一层神经网络：输入为`x`，参数为`w`、`b`，一个损失函数。定义如下：
```python
import torch

x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b  # 计算x*w+b ==> z

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```
## Tensors, Functions and Computational graph
上面那段代码定义的是这样的一个**计算图**（`x * w + b ==> z`）：
> 计算图就是把一个做计算的式子用图来进行表示，这个图就称作“计算图”。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419112157151.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
在这个计算图中，`w`和`b`在初始化时设置了`requires_grad`，这代表它们需要进一步优化。

> **注意：** 可以在创建张量时设置`requires_grad`属性；也可以之后通过` x.requires_grad_(True)`(x是一个张量实例对象)进行设置。

我们应用于张量以构造计算图的函数实际上是[`Function`类](https://pytorch.org/docs/stable/autograd.html#function)的对象。该对象知道如何在正向传播上计算该函数，以及如何在反向传播步骤中计算该函数的**导数(derivative)**。

反向传播函数的应用存储在张量的`grad_fn`属性中：
```python
print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419120601893.png)
## Computing Gradients
为了优化神经网络中的参数权重，我们需要针对参数计算损失函数的导数，也就是说，我们需要在x、y固定的情况下计算：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419121324844.png)
为了计算这些导数，我们调用`loss.backward()`函数，然后从`w.grad`和`b.grad`中检索值：

> **注意：** 
> - 只有计算图中设置了`requires_grad=True`的叶节点才有`grad`属性。【If `x` is a Tensor that has `x.requires_grad=True` then `x.grad` is another Tensor holding the gradient of `x` with respect to some scalar value.】
> - 在一张图上只能执行一次`backward()`来计算梯度。如果需要多次执行，我们需要在执行`backward()`时传入参数`retain_graph=True`。
```python
loss.backward()
print(w.grad)
print(b.grad)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419121813508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
## Disabling Gradient Tracking
默认情况下，所有`requires_grad = True`的张量都在跟踪计算历史并支持**梯度计算(gradient computation)**。

但是，在有些情况下，我们想要**禁用梯度计算**：
1. 已经训练好了模型并只想将其应用于某些输入数据时。也就是说，我们只需要对神经网络做前向计算(do forward computations )。
2. 要将神经网络中的某些参数标记为冻结参数(frozen parameters)。 这是微调预训练网络的非常常见的情况。
3. 仅在进行正向传递时可以加快计算速度，因为在不跟踪梯度的张量上进行计算会更有效。


禁用梯度计算的几种方法如下。

方法一：将计算式代码写在`torch.no_grad()`块内：
```python
import torch

x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b  # 默认情况下，z计算出来就是requires_grad=True的张量
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419124811479.png)
方法二：张量调用`detach()`方法：

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```
## More on Computational Graphs
从概念上讲，autograd在由Function对象组成的有向无环图（DAG）中记录数据（张量）和所有已执行的操作（以及由此产生的新张量）。在此DAG中，叶结点是输入张量，根结点是输出张量。通过从根到叶跟踪该图，可以使用**导数的链式法则**自动计算梯度。

在前向传递中，`autograd`同时执行两项操作：
- 运行请求的操作以计算结果张量；
- 在DAG中维护操作的梯度函数。

当在DAG根目录上调用`.backward()`时，反向传播开始。然后`autograd`执行以下操作：
- 计算每个`.grad_fn`的梯度；
- 将它们累积在各自的张量的`.grad`属性中；
- 使用链式法则，一直传播到叶张量。

## Optional Reading: Tensor Gradients and Jacobian Products
在许多情况下，我们具有标量损失函数，并且需要针对某些参数计算梯度。但是，在某些情况下，输出函数是任意张量。在这种情况下，PyTorch允许您计算所谓的雅可比积，而不是实际的梯度。
-- --

```python
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
```
注意，当我们第二次使用相同的参数调用`backward()`时，导数的值是不同的。发生这种情况是因为PyTorch在进行向后传播时会**累积梯度**，即将计算出的梯度值添加到计算图所有叶节点的grad属性中。如果要计算适当的梯度，则需要先将`grad`属性清零。

以前我们是在没有参数的情况下调用`backward()`函数。这本质上等效于`backward(torch.tensor(1.0))`，这是在标量值函数（例如神经网络训练中的损失）的情况下计算梯度的一种有用方法。
-- --
[torch.autograd API](https://pytorch.org/docs/stable/autograd.html#function)
[Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

# Autograd mechanics
> https://pytorch.org/docs/stable/notes/autograd.html#
## Excluding subgraphs from backward
每个Tensor都有一个标志：`requires_grad`，表示是否需要计算梯度。

如果某个输入是需要梯度的，那么它的输出自动就会存在梯度；只有所有的输入都不需要梯度，输出就也不需要。在所有张量都不要求梯度的子图中，永远不会执行反向计算。代码示例：

```python
import torch

x = torch.randn(5, 5)   # requires_grad=False by default
y = torch.randn(5, 5)   # requires_grad=False by default
z = torch.randn((5, 5), requires_grad=True)

a = x + y
print(a.requires_grad)

b = a + z
print(b.requires_grad)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210505162444371.png)
设置了`requires_grad=False`的参数称作**冻结参数（ frozen parameters)**。以下两种情况，通常会冻结参数：

- 事先知道不需要这些参数的梯度；
- 微调预训练的网络。

在微调中，我们冻结了大部分模型，通常仅修改分类器层以对新标签进行预测。示例：

```python
import torchvision
from torch import optim, nn

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# 在具有10个标签的新数据集上微调模型。
# 在此模型中，充当分类器的是最后一个线性变换层
# 下面用一个新的线性层替换原来的层（默认是requires_grad=True）
model.fc = nn.Linear(512, 10)

# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```
## How autograd encodes the history
Autograd是**反向自动分化系统**。从概念上讲，autograd会记录一个图形，记录执行操作时创建数据的所有操作，从而为您提供一个有向无环图，其叶子为输入张量，根为输出张量。通过从根到叶跟踪该图，可以使用链式法则自动计算梯度。

在内部，autograd将该图表示为Function对象（真正的表达式）的图，调用`apply()`可以计算评估该图的结果。在计算**前向传播(forwards pass)**时，autograd同时执行所需的计算，并建立一个表示计算梯度的函数的图形（每个torch.Tensor的`.grad_fn`属性是此图形的入口点）。完成前向传播后，我们在后向传播中评估此图以计算梯度。

需要注意的重要一点是，每次迭代都会从头开始重新创建图形，这正是允许使用任意Python控制流语句的原因，该语句可以在每次迭代时更改图形的整体形状和大小。

## In-place operations with autograd
在autograd中支持**就地操作(in-place operations)** 很困难，并且在大多数情况下，我们不鼓励使用它们。Autograd积极的缓冲区释放和重用使其非常高效，并且在极少数情况下，就地操作实际上会显著降低内存使用量。除非您在高内存压力下进行操作，否则可能永远不需要使用它们。

限制就地操作的适用性的主要原因有两个：
- 就地操作可能会覆盖计算梯度所需的值。
- 实际上，每个就地操作都需要实现重写计算图。异地版本(Out-of-place versions)仅分配新对象并保留对旧图的引用，而就地操作则需要更改表示此操作的`Function`的所有输入的创建者。这可能很棘手，特别是如果有许多张量引用同一存储（例如通过索引或转置创建的张量），并且如果修改的输入的存储被任何其他张量引用，则就地函数实际上会引发错误。

就地操作的正确性检查：每个张量都有一个版本计数器，每次在任何操作中被标记为脏时(it is marked dirty)，该计数器都会增加。当函数保存任何张量以供后向传播时，也会保存其包含Tensor的版本计数器。访问`self.saved_tensors`后，将对其进行检查，如果该值大于保存的值，则会引发错误。这样可以确保如果您使用的是就地函数并且没有看到任何错误，则可以确保计算出的梯度是正确的。

## Multithreaded Autograd
autograd引擎负责运行计算向后传播所需的所有向后操作。本节将描述所有可以帮助您在多线程环境中充分利用它的细节。（这仅与PyTorch 1.6+相关，因为先前版本中的行为有所不同）。

用户可以使用多线程代码来训练他们的模型（例如Hogwild训练），并且不会阻塞并发的反向计算，示例代码可以是：
```python
import threading
import torch


# 1. 定义一个用来训练的函数
def train_fn():
    x = torch.ones(5, 5, requires_grad=True)
    # forward
    y = (x + 3) * (x + 4) * 0.5
    # backward
    y.sum().backward()
    # potential optimizer update


# 2. 使用多线程的方式来调用训练函数
threads = []
for _ in range(10):
    p = threading.Thread(target=train_fn, args=())
    p.start()
    threads.append(p)

for p in threads:
    p.join()
```

### Concurrency on CPU
当您通过python或C++ API在CPU上的多个线程中运行`backward()`或`grad()`时，您期望看到额外的并发性，而不是在执行期间按特定顺序序列化所有反向调用（在PyTorch1.6之前的行为）。
### Non-determinism
如果您正在多个线程上同时使用共享输入并发调用`back()`（即Hogwild CPU训练）。由于参数是在线程之间自动共享的，因此，在两个线程之间进行反向调用时，梯度累积可能变得不确定，因为两个反向调用可能会访问并尝试累积相同的`.grad`属性。

但这是预期的模式，如果您使用多线程方法来驱动整个训练过程，但使用共享参数，则使用多线程的用户应牢记线程模型，并应期望会发生这种情况。用户可以使用functional  API `torch.autograd.grad()`来计算梯度，而不必使用`backward()`来避免不确定性。

### Graph retaining
如果在线程之间共享autograd图的一部分，即运行前向单线程的第一部分，然后在多个线程中运行第二部分，则图的第一部分将被共享。在这种情况下，不同的线程在同一图上执行`grad()`或`backward()`可能会出现一个线程动态销毁该图的问题，而另一线程在这种情况下将崩溃。Autograd将错误地向用户发出错误，类似于两次调用`backward()`时，都没有`retain_graph = True`，并让用户知道他们应该使用`retain_graph = True`。

### Thread Safety on Autograd Node
由于Autograd允许调用线程驱动其后向执行以实现潜在的并行性，因此重要的是，我们必须确保GraphGraphic的部分/全部与并行向后并行处理在CPU上的线程安全。

由于GIL，自定义Python `autograd.function`自动就是线程安全的。对于内置的C ++ Autograd节点（例如AccumulateGrad，CopySlices）和自定义`autograd :: Function`，Autograd Engine使用线程互斥锁来保护可能具有状态写入/读取的autograd节点上的线程安全。

### No thread safety on C++ hooks
Autograd依靠用户编写线程安全的C++挂钩。如果要在多线程环境中正确应用该钩子，则需要编写适当的线程锁定代码以确保该钩子是线程安全的。

# Optimization
训练模型是一个反复的过程；在每次迭代（称为**epoch**）中，模型都会对输出进行猜测，在其猜测（损失）中计算误差，收集误差相对于参数的导数，并使用**梯度下降( gradient descent)** 来优化参数。
## Hyperparameters
**超参数**是**可调整的**参数，可让您控制模型优化过程。不同的超参数值可能会影响模型训练和收敛速度。

我们定义以下用于训练的超参数：

- **Number of Epochs** - 遍历数据集的次数
- **Batch Size** - 模型在每个时期中看到的数据样本数
- **Learning Rate** - 每个批次/时间段要更新多少模型参数。较小的值会导致学习速度变慢，而较大的值可能会导致训练期间出现无法预测的行为。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


learning_rate = 1e-3
batch_size = 64
epochs = 5
```
## Optimization Loop
设置超参数后，我们便可以使用优化循环来训练和优化模型。优化循环的每次迭代都称为一个**epoch(时期)**。

每一个epoch包含两个主要部分：

- **The Train Loop** - 遍历训练数据集并尝试收敛到最佳参数。
- **The Validation/Test Loop** - 遍历测试数据集以检查模型性能是否正在改善。

### Loss Function
当提供一些训练数据时，我们未经训练的网络很可能无法给出正确的答案。**损失函数**衡量的是*获得的结果与目标值的不相似程度*，这是我们在训练过程中**要最小化**的损失函数。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。

常见的损失函数包括用于回归任务的`nn.MSELoss`（均方误差）和用于分类的`nn.NLLLoss`（负对数似然）。 `nn.CrossEntropyLoss`组合了`nn.LogSoftmax`和`nn.NLLLoss`。

我们将模型的输出logits传递给`nn.CrossEntropyLoss`，这将对logits进行归一化并计算预测误差。

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```
### Optimizer
优化是调整模型参数以减少每个训练步骤中模型误差的过程。优化算法定义了该过程的执行方式（在本例中，我们使用**随机梯度下降法**【Stochastic Gradient Descent】）。所有优化逻辑都封装在**优化器**`optimizer`对象中。

> PyTorch中有许多优化器，比如ADAM、RMSProp、SGD，适用于不同类型的模型和数据。

我们通过注册需要训练的模型参数并传入`learning_rate`超参数来初始化优化器：
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
在训练循环中，优化过程分为三个步骤：
1. 调用`optimizer.zero_grad()`重置模型参数的梯度。默认情况下，**梯度**会**累加**；为了防止重复计算，我们在每次迭代时将它们显式清零。
2. 通过调用`loss.backwards()`向后传播预测损失。PyTorch deposits the gradients of the loss w.r.t. each parameter.
3. 有了梯度后，我们将调用`optimizer.step()`来通过向后传递中收集的梯度来调整参数。

## 完整代码
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

from data.autograd import training_data, test_data

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
# print(model)


learning_rate = 1e-3
batch_size = 64
epochs = 5


# 遍历优化代码
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 根据测试数据评估模型的性能
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()

# 初始化优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10  # 设置遍历数据集的次数
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

-- --
[Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
[torch.optim](https://pytorch.org/docs/stable/optim.html#module-torch.optim)
## [Warmstart Training a Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html)

在转移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见方案。利用经过训练的参数，即使只有少数几个可用的参数，也将有助于**热启动**训练过程，并希望与从头开始训练相比，可以更快地收敛模型。

无论是从缺少某些键的部分`state_dict`加载，还是要使用比要加载的模型更多的键加载`state_dict`，都可以在`load_state_dict()`函数中设置参数`strict=False`，以忽略不匹配项键。

下面尝试使用其他模型的参数来热启动模型。

```python
'''
Steps:
    1. Import all necessary libraries for loading our data
    2. Define and intialize the neural network A and B
    3. Save model A
    4. Load into model B
'''
import torch
import torch.nn as nn
import torch.optim as optim


class NetA(nn.Module):
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

netA = NetA()

class NetB(nn.Module):
    def __init__(self):
        super(NetB, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

netB = NetB()


# Save model A
PATH = "model.pt"  # 模型存储的位置
torch.save(netA.state_dict(), PATH)

# Load into model B
'''
如果要将参数从一层加载到另一层，但某些键不匹配，
只需更改要加载的state_dict中参数键的名称，以匹配要加载到的模型中的键。
'''
netB.load_state_dict(torch.load(PATH), strict=False)
```

