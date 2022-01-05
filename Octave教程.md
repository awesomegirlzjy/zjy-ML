# 3 Octave 教程

## 3.1 基本操作

基本的数学运算：

![image-20210921153412721](E:\md笔记\PyTorch\images\image-20210921153412721.png)

逻辑运算：

![image-20210921153444989](E:\md笔记\PyTorch\images\image-20210921153444989.png)

![image-20210921153608802](E:\md笔记\PyTorch\images\image-20210921153608802.png)

![image-20210921153744989](E:\md笔记\PyTorch\images\image-20210921153744989.png)

变量：

![image-20210921154104577](E:\md笔记\PyTorch\images\image-20210921154104577.png)

![image-20210921154119507](E:\md笔记\PyTorch\images\image-20210921154119507.png)

![image-20210921154048502](E:\md笔记\PyTorch\images\image-20210921154048502.png)

![image-20210921154208163](E:\md笔记\PyTorch\images\image-20210921154208163.png)

![image-20210921154950218](E:\md笔记\PyTorch\images\image-20210921154950218.png)

格式化输出：

![image-20210921155221415](E:\md笔记\PyTorch\images\image-20210921155221415.png)

![image-20210921155501032](E:\md笔记\PyTorch\images\image-20210921155501032.png)

向量和矩阵：

![image-20210921155742927](E:\md笔记\PyTorch\images\image-20210921155742927.png)

![image-20210921160021152](E:\md笔记\PyTorch\images\image-20210921160021152.png)

![image-20210921160122242](E:\md笔记\PyTorch\images\image-20210921160122242.png)

![image-20210921161213417](E:\md笔记\PyTorch\images\image-20210921161213417.png)

![image-20210921161402308](E:\md笔记\PyTorch\images\image-20210921161402308.png)

![image-20210921161559490](E:\md笔记\PyTorch\images\image-20210921161559490.png)

![image-20210921162118794](E:\md笔记\PyTorch\images\image-20210921162118794.png)

![image-20210921162247162](E:\md笔记\PyTorch\images\image-20210921162247162.png)

## 3.2 移动数据

![image-20210921163101471](E:\md笔记\PyTorch\images\image-20210921163101471.png)

![image-20210921163129698](E:\md笔记\PyTorch\images\image-20210921163129698.png)

![image-20210921163147336](E:\md笔记\PyTorch\images\image-20210921163147336.png)

![image-20210921163623979](E:\md笔记\PyTorch\images\image-20210921163623979.png)

### 读取和存储文件数据

假设有两个存放数据的文件**featuresX.dat** 和**priceY.dat**：

- **featuresX**文件如下图，是一个含有两列数据的文件，数据集中有47行，第一列代表房子面积，第二列代表卧室数量。

- **priceY**文件是训练集中的价格数据。

使用`load`命令加载上面两个数据文件：

```matlab
>> load featuresX.dat		% 加载featuresX.dat文件
>> load priceY.dat			% 加载priceY.dat文件
>> load('featuresX.dat')	% 加载featuresX.dat文件
>> load('priceY.dat')		% 加载priceY.dat文件
>>
>> who  % 用来显示Octave工作空间中的所有变量
>> featuresX  % 这将能够显示出来featuresX.dat文件中的所有数据
>> size(featuresX)  % 结果为(47, 2)
>> whos  % 显式出所有变量以及其更详细的信息，如变量的维数、字节数、数据类型
>> clear(featuresX) % 删除某个变量，这里删除了featuresX
>> clear  % 删除所有的变量
>>
>> V = priceY(1:10) % 表示将向量priceY的前10个元素存入V中
>> save hello.mat v % 将变量V存储到一个名为hello.mat的文件【.mat格式会把数据按照二进制形式压缩存储，占用空间小】
>> save hello.txt v -ascii % 把数据存成一个文本文档（用ascii码编码的文档）
```

### 操作数据/矩阵

![image-20210922101131487](E:\md笔记\PyTorch\images\image-20210922101131487.png)

![image-20210922101150148](E:\md笔记\PyTorch\images\image-20210922101150148.png)

![image-20210922101831637](E:\md笔记\PyTorch\images\image-20210922101831637.png)

![image-20210922102028636](E:\md笔记\PyTorch\images\image-20210922102028636.png)

![image-20210922102235212](E:\md笔记\PyTorch\images\image-20210922102235212.png)

![image-20210922102515677](E:\md笔记\PyTorch\images\image-20210922102515677.png)

![image-20210922103023270](E:\md笔记\PyTorch\images\image-20210922103023270.png)

> `C = [A B]`等价于`C = [A, B]`

![image-20210922104034229](E:\md笔记\PyTorch\images\image-20210922104034229.png)

## 3.3 计算数据

![image-20210922110429048](E:\md笔记\PyTorch\images\image-20210922110429048.png)

![image-20210922110446610](E:\md笔记\PyTorch\images\image-20210922110446610.png)

![image-20210922110509637](E:\md笔记\PyTorch\images\image-20210922110509637.png)

![image-20210922111453810](E:\md笔记\PyTorch\images\image-20210922111453810.png)

![image-20210922112413255](E:\md笔记\PyTorch\images\image-20210922112413255.png)

![image-20210922112611153](E:\md笔记\PyTorch\images\image-20210922112611153.png)

![image-20210922112935772](E:\md笔记\PyTorch\images\image-20210922112935772.png)

![image-20210922214531643](E:\md笔记\PyTorch\images\image-20210922214531643.png)

![image-20210922214624692](E:\md笔记\PyTorch\images\image-20210922214624692.png)

![image-20210922215009838](E:\md笔记\PyTorch\images\image-20210922215009838.png)

![image-20210922215105551](E:\md笔记\PyTorch\images\image-20210922215105551.png)

![image-20210922231855026](E:\md笔记\PyTorch\images\image-20210922231855026.png)

```matlab
>> % 求A矩阵的所有元素中最大的那个：
>> max(max(A)) % 法一
>> max(A(:))   % 法二
>> % 原理都是先变成一个向量，再用max函数。
```

![image-20210922231620882](E:\md笔记\PyTorch\images\image-20210922231620882.png)

![image-20210922215414602](E:\md笔记\PyTorch\images\image-20210922215414602.png)

- **magic函数**：返回一个矩阵，称为魔方阵或幻方 (**magic squares**)，它们具有以下这样的数学性质——它们所有的行、列、对角线加起来都等于相同的值。通常只是用来生成一个矩阵。实例：

![image-20210922215748072](E:\md笔记\PyTorch\images\image-20210922215748072.png)

```matlab
>> sum(E,1) % E矩阵每一列对应的和
>> sum(E,2) % E矩阵每一行对应的和
>> sum(sum(E .* eye(3)))  % E矩阵主对角线元素的和【eye(3)表示构造一个 3×3 的单位矩阵】
>> sum(sum(E .* flipud(eye(3)))) % E矩阵次对角线元素的和【flipud函数对矩阵垂直翻转】
```

![image-20210922233714469](E:\md笔记\PyTorch\images\image-20210922233714469.png)

![image-20210922230754178](E:\md笔记\PyTorch\images\image-20210922230754178.png)



![image-20210922231046798](E:\md笔记\PyTorch\images\image-20210922231046798.png)

![image-20210922231231941](E:\md笔记\PyTorch\images\image-20210922231231941.png)

## 3.4 数据可视化

准备数据：

```matlab
>> t = [0:0.01:0.98];
>> y1 = sin(2*pi*4*t);
```

绘制正弦函数：

```matlab
>> plot(t, y1)
```

![image-20210923083421600](E:\md笔记\PyTorch\images\image-20210923083421600.png)

绘制余弦函数：

```matlab
>> y2 = cos(2*pi*4*t);
>> plot(t, y2)
```

![image-20210923083634930](E:\md笔记\PyTorch\images\image-20210923083634930.png)

将两图绘制在同一图里：

```matlab
>> plot(t, y1)
>> hold on;
>> plot(t, y2, 'r')  % r表示曲线颜色
>> xlabel('time')
>> ylabel('value')
>> legend('sin', 'cos') % 图例
>> title('myplot')
>>
>> print -dpng 'myplot.png' % 将图保存为一个png文件
>> close % 关闭图像
>>
>> help plot % 查看相关帮助
```

为图像编号：

```matlab
>> figure(1);
>> plot(t, y1);
>> figure(2);
>> plot(t, y2);
```

![image-20210923085508645](E:\md笔记\PyTorch\images\image-20210923085508645.png)

将两个图像绘制在同一张图里：

```matlab
>> subplot(1,2,1)  % 将绘图区分为一个1*2的格子,并使用第一个格子
>> plot(t, y1)
>> subplot(1,2,2)  % 将绘图区分为一个1*2的格子,并使用第二个格子
>> plot(t, y2)
>> close
```

![image-20210923085845897](E:\md笔记\PyTorch\images\image-20210923085845897-16323587558681.png)

改变轴的刻度：

```matlab
>> plot(t, y1)
>> axis([0.5 1 -1 1]) % 设置横轴的范围为0.5到1，纵轴的范围为-1到1
```

![image-20210923091737035](E:\md笔记\PyTorch\images\image-20210923091737035.png)

清除一幅图像：

```matlab
clf
```

![image-20210923091913352](E:\md笔记\PyTorch\images\image-20210923091913352.png)

```matlab
>> A = magic(5);
>> imagesc(A)  % 将绘制一个5*5的彩色格图，不同的颜色对应A矩阵中的不同值
>> imagesc(A), colorbar, colormap gray  % 同时执行三个命令。将生成一个灰度分布图，并在右边加入一个颜色条，不同深浅的颜色对应不同的值
```

![image-20210923095223080](E:\md笔记\PyTorch\images\image-20210923095223080.png)

![image-20210923095518958](E:\md笔记\PyTorch\images\image-20210923095518958.png)

## 3.5 循环控制语句、自定义函数

### for循环

```matlab
>> v = zeros(10, 1);
>> for i = 1 : 10,
   		v(i) = 2 ^ i;
   end;
>> v

v =

           2
           4
           8
          16
          32
          64
         128
         256
         512
        1024

```

上面的for循环还可以写成：

```matlab
>> indices = 1:10;
>> for i = indices,
   		v(i) = 2 ^ i;
   end;
```

### while循环

```matlab
>> i = 1;
>> while i <= 5,
       v(i) = 100;
       i = i + 1;
   end;
>> v

v =

         100
         100
         100
         100
         100
          64
         128
         256
         512
        1024

>> i = 1;
>> while true,
        v(i) = 999;
        i = i + 1;
        if i == 6,
            break;
        end;
    end;
>> v

v =

         999
         999
         999
         999
         999
          64
         128
         256
         512
        1024
```

### if-else 循环

```matlab
>> v(1) = 2;
>> if v(1) == 1,
        disp('The value is 1.');
   elseif v(1) == 2,
   		disp('The value is 2.');
   else
        disp('The value is not 1 or 2.');
   end;
   
The value is 2.
```

### 函数

在Octave环境下定义函数，需要新建一个文件，并用函数名对其命名，然后以`.m`为文件后缀名。假设桌面有名为`squarethisnumber.m`的文件，内容如下：

```matlab
function y = squareThisNumber(x)
y = x ^ 2;
```

这表示：函数体为y = x ^ 2;，函数的参数为x，返回值为y。调用此函数：

```matlab
>> cd C:\Users\awesome\Desktop % 首先需要进入到该函数所在的目录下
>> squareThisNumber(5)
```

下面定义一个**可以返回多个值**的函数，文件命名为`SquareAndCubeThisNumber.m`：

```matlab
function [y1, y2] = squareAndCubeThisNumber(x)
y1 = x ^ 2;
y2 = x ^ 3;
```

下面我们想要定义一个用来计算损失函数 $J(\theta)$的函数，计算不同$\theta$值所对应的代价函数值$J$。这个函数如下：

```matlab
function J = costFunctionJ(X, Y, theta)

% X is the "design matrix" containing our trainging examples.
% Y is the class labels.

m = size(X, 1); % number of traing examples
predictions = X * theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions - y) .^ 2; % squared errors

J = 1/(2*m) * sum(sqrErrors);
```

调用此函数：

```matlab
>> X = [1 1; 1 2; 1 3];
>> Y = [1; 2; 3];
>> theta = [0; 1];
>> J = costFunctionJ(X, y, theta)  % 输出结果将为0
```

> 为何上面$J$输出为0？注意到函数的预测方式是predictions = X * theta，做出相应运算后，恰好 predictions 与 Y 是完全一致的，因此损失为0. 可以更改$\theta$的值，再进行结算，$J$就不会为0了。

## 3.6 向量化

这是一个常见的线性回归假设函数：${{h}_{\theta}}(x)=\sum_{j=0}^n {\theta}_j{x}_j$ =$\theta^Tx$.

其中：

![image-20210923181153890](E:\md笔记\PyTorch\images\image-20210923181153890.png)

$\sum_{j=0}^n {\theta}_j{x}_j%$是未向量化的式子，代码实现为：

```matlab
prediction = 0.0;
for j = 1:n+1,  % 下标是从1开始
	prediction = prediction + theta(j) * x(j);
end;
```

$\theta^Tx$是向量化后的式子，代码实现为：

```matlab
prediction = theta' * x;
```

>  总结：使用内置的各种线性代数库会使程序更高效。

向量化的方法在其他编程语言中同样可以实现。下面看一个C++中的向量化的例子。

```cpp
// 未向量化
double prediction = 0.0;
for (int j = 0; j <= n; j++)
    prediction += theta[j] * x[j];

// 向量化
double prediction = theta.transpose() * x;
```

# 