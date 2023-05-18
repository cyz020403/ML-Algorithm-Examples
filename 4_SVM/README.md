# 4_SVM_Report

## 任务描述

此次实验使用支持向量机（Support Vector Machine，SVM）做鸢尾花分类任务。

鸢尾花数据集(Iris Dataset)是机器学习中经典的数据集之一，用于分类和聚类问题。该数据集包含了150个样本，每个样本有四个特征：花萼(sepal)长度、花萼宽度、花瓣(petal)长度和花瓣宽度，同时每个样本都被标记为三个类别之一：Setosa、Versicolour 或 Virginica。

其数据格式类似于：

| sepal_length | sepal_width | petal_length | petal_width | class       |
| ------------ | ----------- | ------------ | ----------- | ----------- |
| 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa |

## 基础知识

出于对实现复杂程度以及数据集的考虑，本次实验实现了线性支持向量机（linear SVM）模型。线性支持向量机通过构建一个最大间隔超平面，将数据集划分为不同的类别。具体来说，该模型使用了拉格朗日对偶性（Lagrange duality）的思想，将原始问题转化为一个对偶问题，通过求解对偶问题来得到最大间隔超平面。

在代码实现中，模型采用了梯度下降的方法，每一次迭代都更新拉格朗日乘子（即alpha），并根据更新后的alpha计算权重向量w和偏置b，最终得到模型。预测时，模型根据超平面的位置将样本划分为正类和负类。

需要注意的是，这种实现思路假设数据线性可分，即没有使用核函数（kernel function）将数据映射到高维空间进行处理。但是，由于数据集规模很小，这种思想仍然在测试数据上取得了 100% 的准确率。

## 具体实现

### 依赖

本次实验采用的依赖如下：

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
```

### SVM 分类器

这部分实现了一个简单的线性 SVM 算法，通过计算拉格朗日乘子 alpha 来确定支持向量，并通过计算超平面的方程来进行分类预测。

#### init 方法

我们定义的 SVM 类有三个参数：学习率 eta、迭代次数 epoch 和随机种子 random_state。它们通过 **init** 方法传入并保存在类中，以便在后续的计算中使用。

```python
class my_SVM:
    def __init__(self, eta=0.001, epoch=1000, random_state=42):
        self.eta = eta
        self.epoch = epoch
        self.random_state = random_state
```

#### fit 方法

在 fit 函数中我们实现模型的训练过程。它接受训练数据 X 和标签 y 作为输入。其中，X 的形状为 [num_samples, num_features]，表示有 num_samples 个样本，每个样本有 num_features 个特征；y 的形状为 [num_samples]，表示每个样本对应的标签。

首先，该函数计算样本数和特征数，并初始化 w、b 和 alpha 三个参数。其中，w 是一个形状为 [num_features] 的向量，表示特征的权重；b 是一个标量，表示截距；alpha 是一个形状为 [num_samples] 的向量，表示每个样本对应的拉格朗日乘子。

其次，该函数开始进行迭代训练。在每一轮迭代中，首先将 y 转化为一个列向量，然后计算 Gram 矩阵 H。Gram 矩阵的每个元素 H[i, j] 表示样本 i 和样本 j 的内积。接着，计算当前 alpha 的梯度 grad，将其加到 alpha 上，并将 alpha 中小于 0 的值设为 0。

迭代结束后，根据计算得到的 alpha，找出支持向量的下标 indexes_sv。对于每个支持向量，更新 w 和 b 的值。

```python
class my_SVM:
    def fit(self, X, y):
        self.num_samples, self.num_features = X.shape
        self.w = np.zeros(self.num_features)
        self.b = 0
        self.alpha = np.zeros(self.num_samples)

        for _ in range(self.epoch):
            y = y.reshape([-1, 1])
            H = np.dot(y, y.T) * np.dot(X, X.T)
            grad = np.ones(self.num_samples) - np.dot(H, self.alpha)
            self.alpha += self.eta * grad
            self.alpha = np.where(self.alpha < 0, 0, self.alpha)

        indexes_sv = [i for i in range(self.num_samples) if self.alpha[i] != 0]
        for i in indexes_sv:
            self.w += self.alpha[i] * y[i] * X[i]
        for i in indexes_sv:
            self.b += y[i] - (self.w @ X[i])
        self.b /= len(indexes_sv)
```

#### predict 方法

最后，我们定义 predict 函数使用训练好的模型对测试数据进行预测。它接受一个形状为 [num_samples, num_features] 的测试数据 X 作为输入，并返回一个形状为 [num_samples] 的预测结果 result。对于每个样本，计算其与超平面的距离（即 w·x+b），并将其作为预测结果。如果距离大于 0，则预测为正类（即 1）；否则预测为负类（即 -1）。

```python
class my_SVM:
    def predict(self, X):
            hyperplane = np.dot(X, self.w) + self.b
            result = np.where(hyperplane > 0, 1, -1)
            print(result)
            return result
```

### 数据加载

构建好模型之后，我们需要加载数据进行训练。首先，我们使用 pandas 库的 read_csv 函数从网络读取鸢尾花数据集，并将其存储在名为 iris 的 DataFrame 中。该数据集包含 150 个样本和 4 个特征，以及每个样本的类别标签，因此，将 DataFrame 中的前 4 列作为特征，最后一列作为标签，将它们分别存储在名为 X 和 y 的数组中。

然后，该代码使用 StandardScaler 类对 X 进行标准化处理，使其各个特征的均值为 0，标准差为 1，这种做法可以消除不同特征的量纲差异，使得算法更加准确和稳定。

除此以外，该代码将 y 中的标签值 0 替换为 -1，这是为了符合线性 SVM 中的假设，即样本的标签只能为 1 或 -1。

最后，该代码使用 train_test_split 函数将数据集划分为训练集和测试集。其中，X_std 和 y 是输入数据，test_size=0.2 表示将 20% 的数据划分为测试集，stratify=y 表示按照 y 中的类别比例对数据进行分层抽样。函数的返回值是 X_train、X_test、y_train 和 y_test，分别表示训练集和测试集的特征和标签。

```python
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

sc = StandardScaler()
X_std = sc.fit_transform(X)
y = np.where(y==0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, stratify=y)
```

### 训练与测试

这部分代码对应 main 方法，即应用上述加载的数据，对模型进行训练和测试。其实现思路为：

首先创建了一个 my_SVM 的实例对象 mysvm，并调用其 fit 方法对训练集 X_train 和 y_train 进行训练。fit 方法实现了 SVM 的训练过程，包括对数据集中的样本进行处理、计算 alpha、计算权重 w 和偏置 b 等步骤，最终得到一个训练好的 SVM 模型。

然后，调用 mysvm 的 predict 方法对测试集 X_test 进行预测，并将预测结果存储在名为 y_pred 的变量中。predict 方法实现了 SVM 的预测过程，通过计算超平面的值来预测样本的标签。

最终，计算预测准确率。预测准确率是指模型在测试集上正确预测样本的比例。具体地，代码使用 np.sum 函数计算预测正确的样本数量，再除以测试集样本总数，得到准确率。最后，代码将准确率打印出来。

```python
mysvm = my_SVM()
mysvm.fit(X_train, y_train)
y_pred = mysvm.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy: ", accuracy)
```

## 运行结果

### 准确率

项目最终的运行准确率为：$100$ %。

```shell
Accuracy:  1.0
```
