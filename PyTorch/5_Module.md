# Build Model
神经网络由对数据执行操作的层/模块组成。神经网络本身就是一个由其他模块（层）组成的模块。这种嵌套结构允许轻松地构建和管理复杂的体系结构。
## Build the neural network
下面的程序构建一个对FashionMNIST数据集中的图片进行分类的神经网络。
```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 在CPU或GPU上训练模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))


# 定义NN
class NeuralNetwork(nn.Module):  # 继承自nn.Module类
    def __init__(self):  # initialize the neural network layers
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

        def forward(self, x):  # 重写forward()——对输入数据进行操作
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


model = NeuralNetwork().to(device)  # 创建NeuralNetwork实例对象，并将其绑定到device（CPU or GPU）
# print(model)

X = torch.rand(1, 28, 28, device=device)  # 传入输入数据：1张28*28大小的图片
logits = model(X)  # 这会使得执行forward()
pred_probab = nn.Softmax(dim=1)(logits)   # 通过将其传递给nn.Softmax模块的实例来获得预测概率
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
结果应该是（官网）：
```
Predicted class: tensor([4], device='cuda:0')
```
但实际我的有报错：
```
Traceback (most recent call last):
  File "E:/Flask/pytorch_test/build_neural_network4.py", line 36, in <module>
    logits = model(X)  # 这会使得执行forward()
  File "E:\virtualenvs\pytorch_test\lib\site-packages\torch\nn\modules\module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "E:\virtualenvs\pytorch_test\lib\site-packages\torch\nn\modules\module.py", line 201, in _forward_unimplemented
    raise NotImplementedError
NotImplementedError
```

## Model Layers
下面对上个程序中(Build the neural network小节)的内容进行详细的讲述（主要是对用到的方法的讲解，不过这些方法在这里其实被称作一个个模型层）：

```python
import torch
from torch import nn

input_image = torch.rand(3, 28, 28)  # 3张大小为28x28的图片
# print(input_image.size())


'''
nn.Flatten():
    1. 初始化一个nn.Flatten层，就可以利用初始化的这个变量把2维图片转换成一个像素值的连续数组。
    2. the minibatch dimension (at dim=0) is maintained
'''
flatten = nn.Flatten()
flat_image = flatten(input_image)  # convert each 2D 28x28 image into a contiguous array of 784 pixel values
# print(flat_image.size())   # 输出结果：torch.Size([3, 28, 28])


'''
nn.Linear():
    1. 线性层模块使用它所存储的权重和偏差(weights and biases)对输入数据应用线性变换。
'''
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
# print(hidden1.size())      # 输出结果：torch.Size([3, 784])


'''
nn.ReLU:
    1. 非线性激活会在模型的输入和输出之间创建复杂的映射。
    2. 在线性变换之后引入非线性，从而帮助神经网络学习各种各样的现象。
    3. 除了使用nn.ReLU引入非线性，还有其他激活方法来引入非线性。
'''
# print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)  # 在线性层之间应用nn.ReLU
# print(f"After ReLU: {hidden1}")


'''
nn.Sequential:
    1. nn.Sequential是一个有序的模块容器，数据按照定义的顺序通过所有模块。
    2. 可以使用nn.Sequential容器快速建立一个网络。
'''
seq_modules = nn.Sequential(    # 使用nn.Sequential容器快速建立一个名为seq_modules的网络
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
# print(logits)


'''
nn.Softmax:
    1. 神经网络的最后一层返回logits(raw values in [-infty, infty])且它会被传递给nn.Softmax模块。
    2. logits被锁放到[0, 1]区间，代表模型对每个类别的预测概率。
    3. 参数dim表示：dim parameter indicates the dimension along which the values must sum to 1.
'''
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)
```

## Model Parameters
对神经网络内部的许多层进行参数化，例如，在训练过程中优化关联权重和偏差。

继承自`nn.Module`类的神经网络的实例对象都可以直接调用`parameters() `或`named_parameters()`方法来看到参数。

```python
import torch
from torch import nn


# 在CPU或GPU上训练我们的模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))


# 定义神经网络
class NeuralNetwork(nn.Module):  # 继承自nn.Module类
    def __init__(self):  # initialize the neural network layers
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

        def forward(self, x):  # 重写forward()——对输入数据进行操作
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits


model = NeuralNetwork().to(device)  # 创建NeuralNetwork实例对象，并将其绑定到device（CPU or GPU）
print("Model structure: ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419103755935.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
-- --
[torch.nn API](https://pytorch.org/docs/stable/nn.html)

# Save & Load Model
> 再看看[这里](https://pytorch.org/docs/stable/notes/serialization.html)的内容——在Python中保存和加载PyTorch张量和模块状态、如何序列化Python模块以便可以在C ++中加载它们。

学习如何通过保存、加载和运行模型预测来保持模型状态。
```python
import torch
import torch.onnx as onnx
import torchvision.models as models
```
## Saving and Loading Model Weights
PyTorch模型将学习到的参数存储在内部状态字典`state_dict`中。这些可以通过`torch.save`方法保留：
```python
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```
要加载模型权重，需要首先创建相同模型的实例，然后使用`load_state_dict()`方法加载参数。
```python
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

> 注意：请确保在推断之前调用`model.eval()`方法，以将**dropout layers**和**batch normalization layers**设置为评估模式，不这样做将产生不一致的推论结果。

## Saving and Loading Models with Shapes
加载模型权重时，我们需要首先实例化模型类，因为该类定义了网络的结构。我们可能希望将此类的结构与模型保存在一起，在这种情况下，我们可以将`model`（而不是`model.state_dict()`）传递给保存函数：
```python
torch.save(model, 'model.pth')
```
加载`model`：
```python
model = torch.load('model.pth')
```
> 注意：该方法在序列化模型时使用Python pickle模块，因此在加载模型时依赖于实际的类定义。

## Exporting Model to ONNX
PyTorch还具有本机ONNX导出支持。但是，鉴于PyTorch执行图的动态性质，导出过程必须遍历执行图以生成持久化的ONNX模型。因此，应将适当大小的测试变量传递到导出例程中（在我们的示例中，我们将创建正确大小的虚拟零张量）：
```python
input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')
```
ONNX模型可以做很多事情，包括在不同的平台和不同的编程语言上运行推理。
有关更多详细信息，我们建议访问[ONNX教程](https://github.com/onnx/tutorials)。

# Neural Networks
> https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html （已学完）

## 理论基础

> Now that you had a glimpse of `autograd`, `nn` depends on `autograd` to define models and differentiate them. An `nn.Module` contains layers, and a method `forward(input)` that returns the `output`.  Neural networks can be constructed using the `torch.nn` package.

典型的神经网络的训练过程：

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using the simplest update rule —— the Stochastic Gradient Descent (SGD): `weight = weight - learning_rate * gradient`


摘要：
- `torch.tensor`：多维数组。支持诸如`backward()`之类的autograd操作，还保留了梯度w.r.t.张量。
- `nn.Module`：神经网络模型。方便的封装参数的方式，并带有将其移动到GPU、导出、加载等的帮助器。
- `nn.Parameter`：张量类型。分配给`Module`的属性时，会自动注册为参数。
- `autograd.Function`：实现autograd操作的前向和后向定义。每个`Tensor`操作都会创建至少一个`Function`节点，该节点连接到创建`Tensor`并对其历史进行编码的函数。

## 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1、Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # forward()需要自己定义，backward()一般不需要再自己定义了
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# print(net)

params = list(net.parameters())   # 获得模型的可学习参数
# print(len(params))
# print(params[0].size())  		  # conv1's .weight


# 2、Loss Function
input = torch.randn(1, 1, 32, 32) # 设定一个输入数据
output = net(input)  		 	  # 自动执行forward()函数,获得预测结果
# print(output)
target = torch.randn(10)     # target是我们给input设定的对应的标签
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()     # nn.MSELoss：computes the mean-squared error between the input and the target.

loss = criterion(output, target)
print(loss)
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


# 3、Backprop
net.zero_grad()  # 将所有参数的梯度缓冲区归零

print('conv1.bias.grad before backward：')
print(net.conv1.bias.grad)  # 反向传播前的梯度

loss.backward()
# out.backward(torch.randn(1, 10))  # 随机张量反向传播

print('conv1.bias.grad after backward：')
print(net.conv1.bias.grad)  # 反向传播后的梯度


# 4、Update the weights
'''
The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
weight = weight - learning_rate * gradient
'''
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # 清空优化器，否则梯度会累加
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()        # 优化（更新权重）
```

## 代码解释

第2部——Loss Function：

if you follow `loss` in the backward direction, using its `.grad_fn` attribute, you will see a graph of computations that looks like this:

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

So, when we call `loss.backward()`, the whole graph is differentiated w.r.t. the neural net parameters, and all Tensors in the graph that have `requires_grad=True` will have their `.grad` Tensor accumulated with the gradient.

For illustration, let us follow a few steps backward:

```
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```