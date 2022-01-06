>源自：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

目标：使用CIFAR10 数据集，训练一个能够识别 ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’图片的神经网络。

> 我自己的总结：先取得数据；然后定义一个神经网络；然后开始训练——先预测（forward），然后得到损失值并进行反向传播（backward），最后进行优化（optimize.step()）。

```python
import torchvision.transforms as transforms


'''
Step1. Load and normalize CIFAR10
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


'''
展示一部分图像（非必须步骤）
'''
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


'''
Step2. Define a Convolutional Neural Network
'''
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


'''
Step3. Define a Loss function and optimizer
'''
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


'''
Step4. Train the network on the training data
'''
for epoch in range(2):  # 在训练集上训练了两遍我们的神经网络
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save our trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


'''
Step5. Test the network on the test data
'''
# 5.1 对部分测试集进行测试
'''
dataiter = iter(testloader)   # get some random training images
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))  # print images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
outputs = net(images)  # 输出ouputs是这10个类别的能量值。哪个类别对应的能量越高，网络就认为该图像属于哪个类别。
_, predicted = torch.max(outputs, 1)  # 应获取最高能量的那个类别
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
'''

# 5.2 网络在整个测试集上的表现
correct = 0
total = 0
with torch.no_grad():  		   # 不需要计算梯度，只是检验神经网络在测试集上的表现
    for data in testloader:
        images, labels = data
        outputs = net(images)  # 传入的参数是图像
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# 5.3 下面看看该网络在哪些类别上表现得好，在哪些类别上表现的不好
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# 打印——神经网络在各个类别上进行预测的准确性
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
```
![在这里插入图片描述](E:\md笔记\PyTorch\images\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70.png)****