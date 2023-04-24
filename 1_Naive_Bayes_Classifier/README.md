# 1_Naive_Bayes_Classifier

## 任务描述

此次实验使用朴素贝叶斯分类器做鸢尾花分类任务。

鸢尾花数据集(Iris Dataset)是机器学习中经典的数据集之一，用于分类和聚类问题。该数据集包含了150个样本，每个样本有四个特征：花萼(sepal)长度、花萼宽度、花瓣(petal)长度和花瓣宽度，同时每个样本都被标记为三个类别之一：Setosa、Versicolour或Virginica。

其数据格式类似于：

| Attribute1 | Attribute2 | Attribute3 | Attribute4 | type        |
| ---------- | ---------- | ---------- | ---------- | ----------- |
| 5.1        | 3.5        | 1.4        | 0.2        | Iris-setosa |

## 基础知识

（本次实验基于朴素贝叶斯模型，关于模型的详细描述见《机器学习》西瓜书 7.3 节：朴素贝叶斯分类器）

朴素贝叶斯分类器的核心思想是通过先验概率和条件概率来计算后验概率，进而判断样本所属的类别。它的“朴素”之处在于假设各个特征之间相互独立，这个假设在实际应用中可能并不完全成立，但朴素贝叶斯分类器仍然被广泛应用，并取得了不错的效果。
$$
P(c \mid \boldsymbol{x})=\frac{P(c) P(\boldsymbol{x} \mid c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^d P\left(x_i \mid c\right)
$$
其中，$c$  表示类别，$\boldsymbol{x}$ 表示一个样本的各个属性，其数学描述为一个向量，向量的每一位表示对应属性的取值。公式的组成为：

- $P(c \mid \boldsymbol{x})$ ：后验概率。
- $P(\boldsymbol{x} \mid c)$ ：条件概率。
- $P(c)$：先验概率。
- $P(\boldsymbol{x})$：证据因子。

由于在朴素贝叶斯分类其中，我们假设各个特征之间相互独立，因此：
$$
P(\boldsymbol{x} \mid c)=\prod_{i=1}^d P\left(x_i \mid c\right)
$$
故满足上述推到过程。其中，$d$ 为属性数目，$x_i$ 为 $\boldsymbol{x}$ 在第 $i$ 个属性上的取值。

由于对于所有类别来说 $P(\boldsymbol{x})$ 取值相同，不影响最终概率的计算，因此朴素贝叶斯分类器的“函数”可以表示如下：
$$
h_{n b}(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^d P\left(x_i \mid c\right)
$$
其中，$h_{n b}(\boldsymbol{x})$ 表示朴素贝叶斯分类器本身，也即对于一个样本 $x$ 朴素贝叶斯分类器的处理过程。其意义在于选取后验概率最大类别，作为朴素贝叶斯分类器的判定结果。

注意，以上是对于朴素贝叶斯分类器基本思想的描述，用于程序的实现足矣；但是，朴素贝叶斯分类器蕴含的思想以及数学基础远不止这些，只依靠这两个公式也很难从直观上理解朴素贝叶斯分类器的思想和工作过程，欲进一步了解，见《机器学习》西瓜书第 7 章的内容。

## 实现过程

此部分通过对代码的解释，描述朴素贝叶斯做鸢尾花分类任务的实现过程。

### 数据加载

```Python
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
```

这段代码从一个 URL 中读取 CSV 文件，并将其加载到 pandas 数据帧中。

- `pd` 是指 Pandas 库。
- `read_csv()` 是 pandas 中的一个函数，可以读取 CSV 文件并将其转换为pandas数据帧。
- `'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'` 是要读取的 CSV 文件的 URL。
- `header=None` 告诉 pandas，CSV 文件没有标题行，因此**它将为数据帧生成默认列名**。

因此，此代码将从UCI机器学习存储库中读取鸢尾花数据集，并将其转换为pandas数据帧，并使用默认列名。

### 数据处理

```Python
train_data = data.sample(frac=0.7, random_state=1)
test_data = data.drop(train_data.index)
```

以上代码是在Python中用Pandas库对数据集进行划分，以便在机器学习算法中进行训练和测试。

`data`是原始的数据集。

`train_data = data.sample(frac=0.7, random_state=1)` 将原始数据集中的随机70%的数据作为训练数据，其余的30%的数据作为测试数据。其中，`frac=0.7` 指定了训练数据占原始数据集的比例，`random_state=1` 是随机种子，确保每次划分数据集的结果都是一样的。

`test_data = data.drop(train_data.index)`从原始数据集中删除训练数据，并将剩余的数据作为测试数据。这是因为在机器学习中，测试数据必须与训练数据互斥，不能使用相同的数据进行训练和测试，以确保模型的泛化能力。

因此，这两行代码完成了将原始数据集分为训练数据和测试数据的过程。

```Python
groups = train_data.groupby(train_data.iloc[:, -1])
```

以后代码是通过将`train_data`数据框按照最后一列的值 (`iloc[:, -1]`) 进行分组，创建一个名为`groups`的 `DataFrameGroupBy` 对象。

`DataFrameGroupBy` 是pandas的一个对象，它表示已根据某些标准拆分的DataFrame对象的集合。在这种情况下，标准是`train_data`数据框的最后一列中的值。具有相同最后一列值的每个数据组将作为`groups`对象中的单独DataFrame存储。

然后可以使用`groups`对象将各个数据组应用于各种聚合或转换函数，这些函数可以对每个数据组分别进行处理。

### 模型实现

#### 先验概率、条件概率计算

这一部分的代码用于计算每个类别的先验概率和每个特征的条件概率。

```Python
priors = {}
conditionals = {}
for group in groups:
    class_name = group[0]
    priors[class_name] = len(group[1]) / len(train_data)
    conditionals[class_name] = {}
    for i in range(len(group[1].columns) - 1):
        feature_name = group[1].columns[i]
        feature_values = group[1][feature_name]
        mean = feature_values.mean()
        std_dev = feature_values.std()
        conditionals[class_name][feature_name] = {'mean': mean, 'std_dev': std_dev}
```

`priors` 和 `conditionals` 表示两个字典，用于存储计算得到的先验概率和每个特征的条件概率。

循环处理每一类别的数据，`priors[class_name] = len(group[1]) / len(train_data)` 计算了当前类别的先验概率。

对于当前类别下样本的每一个属性，统计属性的名字并计算属性取值的平均值和方差。这里并没有完整计算出条件概率，而是转为计算在某一类别的先验条件下，$\boldsymbol{x}$ 中各个属性的取值平均值与方差。

#### 定义朴素贝叶斯分类器方法

这一部分的代码用于定义朴素贝叶斯分类器方法。

```Python
def predict_class(sample, priors, conditionals):
    probabilities = {}
    for class_name in priors:
        probabilities[class_name] = priors[class_name]
        for i in range(len(sample)):
            feature_name = data.columns[i]
            x = sample[i]
            mean = conditionals[class_name][feature_name]['mean']
            std_dev = conditionals[class_name][feature_name]['std_dev']
            probabilities[class_name] *= (1 / (math.sqrt(2 * math.pi) * std_dev)) * \
                                            math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    predicted_class = max(probabilities, key=probabilities.get)
    return predicted_class
```

在此之前，我们已经“看过”训练集上的数据，并得到了先验概率和每个特征的条件概率，这个方法的作用是根据之前计算的结果，对传入的测试数据进行分析判判断，最终指定样本的类别。

此方法传入了三个数据：

- sample：某一个测试样本，实际存储的数据结构为一个向量，表示当前样本在各个属性上的取值。
- priors：计算得到的先验概率 $P(c)$。
- conditionals：每个特征的条件概率，实际存储的是在某一类别的先验条件下，$\boldsymbol{x}$ 中各个属性的取值平均值与方差。形式化的表示为：`conditionals[class_name][feature_name] = {'mean': mean, 'std_dev': std_dev}`。

定义 `probabilities` 用于存储当前测试样本属于每一个类别的后验概率。循环计算属于每一个类别的后验概率，此概率由以下公式计算：
$$
P(c \mid \boldsymbol{x})=P(c) \prod_{i=1}^d P\left(x_i \mid c\right)
$$
先验概率 $P(c)$ 存储在字典 `priors` 中。而对于条件概率，上文已经阐述过，`conditionals[class_name][feature_name]` 并没有完整计算出条件概率，而是转为计算在某一类别的先验条件下，$\boldsymbol{x}$ 中各个属性的取值平均值与方差。此处需要进一步转换为条件概率。

条件概率的计算比较复杂，回顾一下条件概率的作用：在朴素贝叶斯分类器中，条件概率指的是某个特征值（或特征向量）在给定类别的前提下的概率。具体地，如果我们要对一个样本进行分类，需要计算该样本属于每个类别的概率，然后选取概率最大的类别作为分类结果。（《机器学习》西瓜书的笔记中有对条件概率更为直观的描述）

在计算条件概率时，我们需要根据样本的特征值和先验知识（例如训练集中已知的各个类别的概率）来计算后验概率，即样本属于某个类别的概率**。对于连续型变量，我们通常使用概率密度函数来计算其条件概率**。而正态分布（也称为高斯分布）是一个常用的连续概率分布，因为它具有很好的性质，且在实际应用中较为常见。

因此，在朴素贝叶斯分类器中，通常假设每个特征值都服从独立的正态分布，然后利用这些分布来计算后验概率。对于一个特征值 $x$，其在某个类别下的条件概率可以表示为：
$$
P(x \mid y=c) = \frac{1}{\sqrt{2\pi}\sigma_c} \exp\left(-\frac{(x - \mu_c)^2}{2\sigma_c^2}\right)
$$
其中 $y=c$ 表示样本属于类别 $c$，$\mu_c$ 和 $\sigma_c$ 分别表示训练集中类别 $c$ 下该特征的均值和标准差，这两个数据已经计算完成并存储在了 `conditionals` 字典中。

这个公式可以解释为：$x$ 在类别 $c$ 下的条件概率等于 $x$ 在均值为 $\mu_c$，标准差为 $\sigma_c$ 的正态分布下的概率密度值。公式中的第一个部分是正态分布的标准化常数，用于保证概率密度函数的积分为 1。第二个部分是正态分布的指数部分，表示了 $x$ 在正态分布下的概率密度值。更为直观的理解为：我们在训练数据集上找到了条件概率所服从的正态分布情况（即正态分布参数），现在我们得到了新的样本，我们认为新得到的样本仍服从此正态分布，并以及计算其条件概率。

以上 Python 代码实现了上述公式中概率的计算过程。

代码的最后两行，选择了最大的后验概率，即认为样本最有可能属于当前类别，得到分类结果。

#### 对测试集进行分类

此部分代码是程序的测试过程，在完成上文先验概率和每个特征的条件概率计算的基础上，调用朴素贝叶斯分类器方法，完成测试集上数据的分类，并计算分类的准确率。

```Python
correct = 0
for i in range(len(test_data)):
    sample = test_data.iloc[i, :-1]
    true_class = test_data.iloc[i, -1]
    predicted_class = predict_class(sample, priors, conditionals)
    if predicted_class == true_class:
        correct += 1
accuracy = correct / len(test_data)
print('准确率为：', accuracy)
```

代码中，循环处理每一个测试样本，并统计分类正确的数量。并最终计算准确率。

## 运行结果

### 运行结果

项目最终的运行准确率为：$91.11$ %。

```shell
# 运行项目
$ python nb.py
# 得到运行结果
准确率为： 0.9111111111111111
```

### 运行结果分析

对于运行正确率的分析，进一步探究数据集的质量。

数据集的说明文件 `iris.names` 中有如下描述：

> ```
> The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
> where the error is in the fourth feature.
> The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
> where the errors are in the second and third features.  
> ```

由于数据集本身只有 150 个样本，因此在进行模型分析时，数据集中的这两个错误值得考虑。
