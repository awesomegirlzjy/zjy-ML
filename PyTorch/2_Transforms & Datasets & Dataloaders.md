# Transforms
作用：数据并不总是以训练机器学习算法所需的最终处理形式出现，使用**transforms**对数据进行处理，使其适于做训练。

转换是常见的图像转换。可以使用[`Compose`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose)将很多转换链接在一起。此外，还有`torchvision.transforms.functional`模块，可对转换进行细粒度的控制，如果您必须构建更复杂的转换管道（例如在分段任务中），这将很有用。

所有转换都接受PIL图像、张量图像或一批张量图像作为输入。**张量图像**是具有(C，H，W)形状的张量，其中C是通道的数量，H和W是图像的高度和宽度。**张量图像批次**是(B，C，H，W)形状的张量，其中B是批次中的许多图像。应用于张量图像批处理的确定性或随机变换会相同地转换该批处理的所有图像。
-- --
所有`TorchVision`数据集都有两个参数：
1. transform：用于修改特征（features）
2. target_transform： 用于修改标签（labels）
```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

'''
1. The FashionMNIST features are in PIL Image format, and the labels are integers.
2. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use ToTensor and Lambda.
'''
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),  # ToTensor()能够把PIL格式的图像或者Numpy的ndarray转换成FloatTensor, 并把图片的像素强度值压缩至[0, 1]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))  # turn the integer into a one-hot encoded tensor(首先定义一个维度为10的全0张量，然后调用scatter_()，)
)
```
-- --
转换函数有很多很多，见[API](https://pytorch.org/vision/stable/transforms.html#)。

# Datasets & Dataloaders

用于处理数据样本的代码可能变得凌乱且难以维护。理想情况下，我们希望将数据集代码与模型训练代码分离，以提高可读性和模块化性。

`torch.utils.data.DataLoader`与`torch.utils.data.Dataset`是PyTorch的两个可处理数据的原语，它们可以使您使用预加载的数据集就像在使用您自己的数据。

- `torch.utils.data.Dataset`：PyTorch提供的内置的高质量数据集，存储了样本及其相应的标签。
- `torch.utils.data.DataLoader`：PyTorch数据加载实用程序的核心。`DataLoader`在数据集周围包装一个可迭代对象，以方便访问样本。

PyTorch提供了特定领域的库，例如`TorchText`，`TorchVision`和`TorchAudio`，所有这些库都包含着许多数据集，这些数据集是`torch.utils.data.Dataset`的子类，它们实现了对特定数据的一些操作。比如`TorchVision`中包括的数据集可以在[这里](https://pytorch.org/vision/stable/datasets.html)看到。


## 实例1
可以看看[这个](https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html)(LOADING DATA IN PYTORCH)。笔记如下：
```python
# Step 1、加载/下载数据时所需要的包
import torch
import torchaudio


# Step 2、从数据集中获取数据
'''
参数解读：
    - root(不可省略，其他的都是可选项): 数据集的地址
    - download：是否从网站下载数据集并保存到root所在路径。
    - transform：在数据上使用transform可让您从数据的原始状态中获取数据，且将其转换为结合在一起的、非正规化的、准备好进行训练的数据。
    - target_transform：A function/transform that takes in the target and >transforms it.
'''
torchaudio.datasets.YESNO(
    root,                   # 有报错
    url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
    folder_in_archive="waves_yesno",
    download=False,
    transform=None,         # 有报错
    target_transform=None
)
# 每个数据都是一个元组(waveform, sample_rate, labels),其中labels为0或1
yesno_data = torchaudio.datasets.YESNO('./', download=True)
# 查看索引为3的数据
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))


# Step 3、Loading the data
'''
    上面获取到数据集后，就必须通过torch.utils.data.DataLoader来传递数据；
    DataLoader结合了数据集和采样器，它返回一个可迭代的数据集对象(训练模型时需要使用这个可迭代的对象)。
'''
data_loader = torch.utils.data.DataLoader(yesno_data, batch_size=1, shuffle=True)


# Step 4、遍历数据
for data in data_loader:  # data_loader中的每个数据项都被转换为一个张量data
    print("Data: ", data)
    print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1]. data[2]))

    # Step 5、可视化数据(这一步不是必须进行的)
    import matplotlib.pyplot as plt

    print(data[0][0].numpy())
    plt.figure()
    plt.plot(waveform.t().numpy())

   break
```

## 实例2
[原址](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)。
### Step1、Loading a Dataset
```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(  # torchvision中的FashionMNIST数据集
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
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417153041650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)

### Step2、Iterating and Visualizing the Dataset
```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    # 使用matplotlib进行可视化
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210419105711684.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
### Step3、Creating a Custom Dataset for your files
自定义数据集类必须实现三个功能：`__init__`, `__len__`, and `__getitem__`. 
```python
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):  
        self.img_labels = pd.read_sql(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)  # the number of samples in our dataset.

    def __getitem__(self, idx):      # 返回数据集中指定索引的那个数据
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # the image’s location on disk
        image = read_image(img_path)          # read_image()把图片转换成tensor
        label = self.img_labels.iloc[idx, 1]  # retrieves the corresponding label from the csv data in self.img_labels
        if self.transform:
            img = self.transform(image)  	  # 调用转换函数
        if self.target_transform:
            label = self.target_transform(label)  # 调用转换函数
        sample = {"image": image, "label": label}
        return sample  # returns the tensor image and corresponding label in a Python dict
```
### Step4、Preparing your data for training with DataLoaders
```python
from dataset1 import training_data, test_data

# loaded that dataset into the Dataloader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```
> 设置参数`shuffle=True`——打乱数据可以减少模型过度拟合。
### Step5、Iterate through the DataLoader
We have loaded that dataset into the `Dataloader` and can iterate through the dataset as needed. Each iteration below returns a batch of `train_features` and `train_labels`(containing `batch_size=64` features and labels respectively).
```python
import matplotlib.pyplot as plt

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```