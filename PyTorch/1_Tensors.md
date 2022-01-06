> ==标量（Scalar）==：一个独立存在的数。标量的运算就是平常做的数字算数。
> ==向量（Vector）==：向量是指一列按顺序排列的元素，习惯用括号将这一列元素括起来，其中的每个元素都由一个索引值来惟一的确定其在向量中的位置。向量中的不同数字还可以用于表示不同坐标轴上的坐标值。
> ==矩阵（Matrix）==：矩阵就是一个二维数组结构。
> ==张量（Tensor）==：若数组的维度超过了二维，我们就可以用张量来表示。
# Tensors

tensors(张量)是一种类似于数组或矩阵的特殊的数据结构。

> [官网](https://pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)：Tensors are similar to NumPay’s ndarrays, ....... 

## tensor的几种初始化方式
```python
import torch
import numpy as np

# 1. 直接使用数据进行初始化
data = [[10,20], [30,40]]
x_data = torch.tensor(data)
# print(x_data)

# 2. Numpy array 转换成 tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# print(np_array)
# print(x_np)

# 3. 从另一个tensor创建新的tensor
x_ones = torch.ones_like(x_data) # etains the properties of x_data
# print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(x_rand)

# 4. 元素值随机或被指定为0/1的tensor
shape = (2, 3,) # 使用shape：用来决定维数
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
# print(rand_tensor)
# print(ones_tensor)
# print(zeros_tensor)
```
## tensor的属性
```python
import torch

t = torch.tensor([[0,1,2], [3,4,5]])
print('数据 = {}'.format(t))
print(f'大小 = {t.size()}')  # 1
print(f"大小 = {t.shape}")   # 2
print(f'维度 = {t.dim()}')
print(f'元素个数 = {t.numel()}')
print(f"Datatype of tensor: {t.dtype}")
print(f"Device tensor is stored on: {t.device}")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210427140813486.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)


## tensor的方法
### 索引、切片
```python
import torch

t = torch.arange(16)
tensor = t.reshape(4, 4)  # 将t编程4行4列的矩阵
print('First row:', tensor[0])
print('First column:', tensor[:, 0])  # ","用来分隔不同维度
print('Last column:', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042714272629.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)

### 扩展、拼接
`torch.cat()`
```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # 参数dim用来指示将这些张量在哪个维度进行拼接
print(t1)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417115430200.png)
`tensor型变量.repeat()`
```python
import torch

t12 = torch.tensor([[5., -9.], [3, 4]])
print(f't12 = {t12}')
t34 = t12.repeat(3, 2)
print(f't34 = {t34}')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210427150828849.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)

### 矩阵相乘
> 三种方法可以计算两矩阵相乘。
```python
tensor = torch.ones(4, 4)
tensor[:, 1] = 0

y1 = tensor @ tensor.T # tensor.T是tensor的转置

y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)
print(y2)
print(y3)
```
### 逐元素乘积
> - 逐元素乘积指的是两个矩阵对应位置上的元素相乘。
> - 三种方法可以计算逐元素乘积。

```python
data1 = [[1,2,3], [4,5,6]]
x_data1 = torch.tensor(data1)

z1 = x_data1 * x_data1

z2 = x_data1.mul(x_data1)

shape = (2, 3,)
rand_tensor = torch.rand(shape)
z3 = torch.rand_like(rand_tensor) # 不能把rand_tensor换成x_data1, 会报错
torch.mul(rand_tensor, rand_tensor, out=z3) 

print(z1)
print(z2)
print(z3)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417124735941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
> 注意：`torch.mul(rand_tensor, rand_tensor, out=z3)` 把rand_tensor换成x_data1, 会报如下错误：
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417125032635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)

以下运算均属于逐元素运算：
```python
import torch

t1 = torch.tensor([[1,2,3], [4,5,6]])
t2 = torch.tensor([[7,8,9], [10,11,12]])
print(t1 + t2)
print(t1 * t2)
print(t1 / t2)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021042716345233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)
### 将单元素的tensor类型转换为Python变量：`item()`
> 将单元素tensor转换成Python变量。
```python
data = [[1,2,3], [4,5,6]]
x_data = torch.tensor(data)

agg = x_data.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021041712575073.png)
### 就地操作：`_` suffix
> **In-place operations** Operations that store the result into the operand are called in-place. They are denoted by a `_` suffix. For example: `x.copy_(y)`, `x.t_()`, will change `x`.
> 官方提醒：In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, **their use is discouraged**.

```python
data = [[1,2,3], [4,5,6]]
tensor = torch.tensor(data)

print(tensor)
tensor2 = tensor.add(5) # 不会改变tensor的值
print(tensor)
print(tensor2)

tensor.add_(5) # 使矩阵中每个元素都加5，且结果值保存到原矩阵
print(tensor)
```
## tensor与numpy数组的转换
> Tensors on the CPU and NumPy arrays can **share their underlying memory locations**, and changing one will change the other.
### Tensor to NumPy array：`tensor变量.numpy()`
```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)  # A change in the tensor reflects in the NumPy array.
print(f"t: {t}")
print(f"n: {n}")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210417132420201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p5X3oxMTEyMQ==,size_16,color_FFFFFF,t_70)

### NumPy array to Tensor： `torch.from_numpy(Numpy变量)`
```python
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)  # Changes in the NumPy array reflects in the tensor.
print(f"t: {t}")
print(f"n: {n}")
```

