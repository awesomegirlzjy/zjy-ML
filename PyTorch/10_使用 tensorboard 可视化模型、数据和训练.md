TensorBoard 是一种用于可视化神经网络训练运行结果的工具，PyTorch 与 TensorBoard 集成。

本教程使用 Fashion-MNIST 数据集说明了它的一些功能，该数据集可以使用 `torchvision.datasets` 读入 PyTorch。

In this tutorial, we’ll learn how to:

1. Read in data and with appropriate transforms (nearly identical to the prior tutorial).

2. Set up TensorBoard.

3. Write to TensorBoard.

4. Inspect a model architecture using TensorBoard.

5. Use TensorBoard to create interactive versions of the visualizations we created in last tutorial, with less code

   - A couple of ways to inspect our training data
   - How to track our model’s performance as it trains
   - How to assess our model’s performance once it is trained.

# 准备数据

> 同7_eg_Training a Classifier.md

```python
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
	[transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
)

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
                                             download=True,
                                             train=True,
                                             transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
                                           download=True,
                                           train=False,
                                           transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# 定义一个包含所有分类的常量
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# 定义一个用来显示图像的辅助函数(used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
```

我们将在该教程中定义一个类似的模型架构，只进行微小的修改以考虑到图像现在是一个通道而不是三个通道和 28x28 而不是 32x32 的事实：

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
```

定义`optimizer`和`criterion`：

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

# 设置TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
# SummaryWriter是将信息写入 TensorBoard 的关键对象
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

请注意，仅此行会创建一个 running/fashion_mnist_experiment_1 文件夹。

# 写入TensorBoard

现在让我们使用 make_grid 将图像写入 TensorBoard - 特别是网格。

```python
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
```

在命令行中分别执行以下两语句：

```python
pip install tensorboard
tensorboard --logdir=runs
```

导航到 http://localhost:6006 应显示以下内容：

![intermediate/../../_static/img/tensorboard_first_view.png](https://pytorch.org/tutorials/_static/img/tensorboard_first_view.png)

但是不知道为什么我的有问题：

![image-20210911190111649](E:\md笔记\PyTorch\images\image-20210911190111649.png)

# 使用 TensorBoard 检查模型

TensorBoard 的优势之一是其可视化复杂模型结构的能力。让我们可视化我们构建的模型。

```python
writer.add_graph(net, images)
wrier.close()
```

现在刷新 TensorBoard 后，您应该会看到一个“Graphs”选项卡，如下所示：

![intermediate/../../_static/img/tensorboard_model_viz.png](https://pytorch.org/tutorials/_static/img/tensorboard_model_viz.png)

继续并双击“Net”以查看其展开，查看构成模型的各个操作的详细视图。 

TensorBoard 有一个非常方便的功能，可以在低维空间中可视化高维数据，例如图像数据；我们接下来会介绍这个。

# 向 TensorBoard 添加“投影仪”

我们可以通过 add_embedding 方法可视化高维数据的低维表示。

```python
# 辅助函数: 从数据集中选择 n 个随机数据点及其相应的标签
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# 选择随机图像及其目标索引
images, labels = select_n_random(trainset.data, trainset.targets)
# 获取每张图片的类别标签
class_labels = [classes[lab] for lab in labels]
# 日志嵌入
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()
```

现在，在 TensorBoard 的“投影仪”选项卡中，您可以看到这 100 张图像——每张都是 784 维的——被投影到 3 维空间中。此外，这是交互式的：您可以单击并拖动以旋转三维投影。最后，一些使可视化更容易查看的提示：选择左上角的“color：label”，以及启用“night mode”，这将使图像更容易看到，因为它们的背景是白色的：

![intermediate/../../_static/img/tensorboard_projector.png](https://pytorch.org/tutorials/_static/img/tensorboard_projector.png)

现在我们已经彻底检查了我们的数据，让我们展示 TensorBoard 如何让跟踪模型训练和评估更清晰，从训练开始。

# 使用 TensorBoard 跟踪模型训练

在前面的例子中，我们简单地每 2000 次迭代打印模型的运行损失。现在，我们将把运行损失记录到 TensorBoard，同时查看模型通过 plot_classes_preds 函数所做的预测。

```python
# 辅助函数：从经过训练的网络和图像列表生成预测和相应的概率
def images_to_probs(net, images):
    output = net(images)
    # 将输出概率转换为预测类别
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

'''
辅助函数：
	1. 使用经过训练的网络以及批次中的图像和标签生成 matplotlib 图，
	   该图显示网络的最高预测及其概率，以及实际标签，并根据预测是否正确为该信息着色。
	2. 使用“images_to_probs”函数。
'''
def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    # 绘制批次中的图像以及预测和真实标签
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:1f}%\n(label: {2})".format(
                                    classes[preds[idx]],
                                    probs[idx] * 100.0,
                                    classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red")
                    )
        return fig
```

最后，让我们使用与上一教程相同的模型训练代码来训练模型，但每 1000 批将结果写入 TensorBoard，而不是打印到控制台；这是使用 add_scalar 函数完成的。

此外，在训练时，我们将生成一张图像，显示模型的预测与该批次中包含的四张图像的实际结果。

```python
running_loss = 0.0
for epoch in range(1):  # 多次循环数据集
    for i, data in enumerate(trainloader, 0):
        # 获取输入；数据是[输入，标签]的列表
        inputs, labels = data
        
        # 将参数梯度归零
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
		running_loss += loss.item()
        # 每循环到1000批次时记录损失
        if i % 1000 == 999:  
            writer.add_scalar('training loss',
                             running_loss / 1000,
                             epoch * len(trainloader) + i)
         # 记录一个 Matplotlib 图，显示模型对随机小批量的预测  
        writer.add_figure('predictions vs. actuals',
                         plot_classes_preds(net, inputs, labels),
                         global_step=epoch * len(trainloader) +i)
        running_loss = 0.0
print('Finished Training')
```

点击“SCALARS”栏，查看在 15,000 次训练迭代中绘制的运行损失：

![intermediate/../../_static/img/tensorboard_scalar_runs.png](https://pytorch.org/tutorials/_static/img/tensorboard_scalar_runs.png)

此外，我们可以查看模型在整个学习过程中对任意批次所做的预测。查看“IMAGES”选项卡并在“predictions vs. actuals”可视化下向下滚动以查看；这向我们展示了，例如，在仅仅 3000 次训练迭代之后，该模型已经能够区分视觉上不同的类别，例如衬衫、运动鞋和外套，尽管它不像后来的训练那样自信：

![intermediate/../../_static/img/tensorboard_images.png](https://pytorch.org/tutorials/_static/img/tensorboard_images.png)

在之前的教程中，我们在训练模型后查看了每个类别的准确率；在这里，我们将使用 TensorBoard 为每个类绘制精确**召回曲线** (good explanation [here](https://www.scikit-yb.org/en/latest/api/classifier/prcurve.html)) 。

# 使用 TensorBoard 评估训练模型

```python
# 1. 在 test_size x num_classes Tensor 中获取概率预测 
# 2. 在 test_size Tensor 中获取预测值 
# 运行大约需要 10 秒
class_probs = []
class_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        class_probs.append(class_probs_batch)
        class_label.append(labels)
        
test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)

# 辅助函数：接收从 0 到 9 的“class_index”并绘制相应的精确召回曲线
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# 绘制所有的 pr 曲线
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_label)
```

您现在将看到一个“PR CURVES”选项卡，其中包含每个类的精确召回曲线。观察发现，在某些类别中，模型的“曲线下面积”接近 100%，而在其他类别中，该面积较低：

![intermediate/../../_static/img/tensorboard_pr_curves.png](https://pytorch.org/tutorials/_static/img/tensorboard_pr_curves.png)

这是 TensorBoard 和 PyTorch 与其集成的介绍。当然，您可以在 Jupyter Notebook 中完成 TensorBoard 所做的一切，但使用 TensorBoard，您可以获得默认交互的视觉效果。