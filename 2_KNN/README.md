# 2_KNN

## 任务描述

此次实验使用 KNN 做鸢尾花分类任务。

鸢尾花数据集(Iris Dataset)是机器学习中经典的数据集之一，用于分类和聚类问题。该数据集包含了150个样本，每个样本有四个特征：花萼(sepal)长度、花萼宽度、花瓣(petal)长度和花瓣宽度，同时每个样本都被标记为三个类别之一：Setosa、Versicolour 或 Virginica。

其数据格式类似于：

| sepal_length | sepal_width | petal_length | petal_width | class       |
| ------------ | ----------- | ------------ | ----------- | ----------- |
| 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa |

## 基础知识

### 整体描述

KNN (k-Nearest Neighbors) 是一种基于实例的非参数化分类和回归方法，用于通过将新的实例与训练集中最相似的k个邻居进行比较来进行预测。

KNN 模型较为简单，其较为直观的理解方式是：

对于分类问题，每个需要判断的样本，在训练数据集中找到距离自己最近的 $k$ 个邻居，根据其邻居的类别以“投票”的方式判断自己的类别。

对于回归问题，则使用训练样本的值进行预测。

### 距离计算

在KNN中，每个实例都由一组特征表示，并且可以使用欧几里得距离或曼哈顿距离等度量来计算实例之间的相似性。

#### 欧几里得距离

一般来说，两个向量间的”距离“指的就是欧几里得距离，两个向量之间的距离可以用它们的范数来定义。一般来说，我们会使用欧几里得范数（也称为 $L^2$ 范数）来计算向量之间的距离。假设有两个 $n$ 维向量 $\mathbf{x}$ 和 $\mathbf{y}$，它们的欧几里得范数分别为：

$$
\|\mathbf{x}\|_2=\sqrt{x_1^2+x_2^2+\dots+x_n^2}\\
\|\mathbf{y}\|_2=\sqrt{y_1^2+y_2^2+\dots+y_n^2}\
$$

这样，两个向量之间的距离（也称为欧几里得距离）就可以表示为它们之间的欧几里得范数差的绝对值：

$$
d(\mathbf x,\mathbf y)=\|\mathbf x-\mathbf y\|_2=\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+\cdots+(x_n-y_n)^2}
$$

可以看出，欧几里得距离的值越小，表示两个向量越接近。

#### 曼哈顿距离

曼哈顿距离（Manhattan Distance）也称为城市街区距离（City Block Distance），它是指从一个点到另一个点沿着网格线走的最短距离。

对于两个n维向量 $x=(x_1, x_2, \ldots, x_n)$ 和 $y=(y_1, y_2, \ldots, y_n)$，它们之间的曼哈顿距离可以用下面的公式表示：

$$
d(x,y) = \sum_{i=1}^{n} |x_i - y_i|
$$

其中， $|x_i - y_i|$ 表示向量 $x$ 和 $y$ 在第 $i$ 维上的差的绝对值。

### 懒惰学习（lazy learning）

值得注意的一点是，KNN算法属于一种“懒惰”学习（lazy learning）算法，也被称为基于实例的学习（instance-based learning）。

相对于“急切”学习（eager learning）算法，KNN不会在训练阶段构建模型或提取特征，而是直接将所有训练实例存储在内存中，并在需要进行预测时，根据新的实例和训练实例之间的距离或相似度来进行分类或回归。

## 具体实现

对于程序具体实现的描述，根据代码实现的由易到难阐述，并在最终的 `main` 方法中进行统一的阐述。

### 数据处理

这部分的代码用于加载鸢尾花数据集，并进行基本的处理。

```python
# load the data
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
```

本方法中的传入参数分别为：

- `filename`：通过 `main` 传入的包含鸢尾花数据集的文件路径以及文件名。
- `split`：表示划分的训练集在整个数据集中的占比。
- `trainingSet`：训练集列表，用于存储划分得到的训练集数据。
- `testset`：测试集列表，用于存储划分得到的测试集数据。

首先，加载文件中的数据，得到的 `dataset` 列表的维度为 (data_length, 5)，data_length 即数据集文件中的鸢尾花样本数量。每个样本中对应的五个数据描述如上文“人物描述”中所示。

此后，循环处理数据集中的每个样本，取前四个具体数据转化为 `float` 类型加载到 `dateset` 列表中。

最终，为了保证数据的随机性，通过生成随机数，确定每一个样本划分到训练集或测试集中，数据量大时可以保证数据划分的比例趋近于设定好的划分比例。

### 距离计算

根据“基础知识”中的描述，我们通过计算两个样本之间的相似度，权衡不同数据样本间的“距离”，也即 KNN 算法中所描述的两个样本间的“距离”。

我们将样本看作向量，有很多方法可以用于计算两个向量间的距离，我们在这里选择了欧几里得距离进行计算。

以下代码实现了两个样本之间欧式距离的计算：

```python
# Calculate distance
def euclideanDistance(instance1, instance2, length):
    distance = 0 
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
```

本方法中的传入参数分别为：

- `instance1`：样本 1。
- `instance2`：样本2。
- `length`：样本中数据的维度，对于鸢尾花数据集，这个数据为 4。

将两个样本看作两个向量，代码中实现了下述欧式距离公式：

$$
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_{i}-y_{i})^{2}}
$$

### 获取最近邻居

在上文中，我们已经可以计算了任意两个样本间的距离，接下来我们希望可以得到每个节点最近的 k 个邻居，并以此作为区分判断每个节点类别的信息。

```python
# Gets the nearest k neighbors
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
```

本方法中的传入参数分别为：

- `trainingSet`：训练数据，已知其类别信息，用于根据自身类别信息对训练数据进行分类。
- `testInstance`：测试数据，对于传入的当前实例，需要找到前 `k` 个“最相似”的训练数据，并返回。
- `k`：指定了对于每个测试数据需要“看”的训练样本数。

在方法中，对于当前的测试数据样本，循环处理训练集数据，并使用 `euclideanDistance` 方法计算当前测试样本到每个训练样本之间的距离，计算的结果存储到 `distances` 列表中，列表的维度长度为 trainset_count。

得到当前测试样本到各个训练样本的所有距离之后，对 `distance` 中的所有距离进行排序，并最终选取前 $k$ 个数据，判断为当前测试样本的邻居，即与其相似的数据。

### 投票法决定类别

在上文中我们已经可以获得每个测试样本最为相似的 `k` 个训练样本，在这一部分，我们要根据训练样本对每个测试样本进行类别的判断。

```python
# The voting law determines categories
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
```

本方法中的传入参数分别为：

- `neighbors`：数据维度为 $(testset\_count, k)$，表示了与每个测试样本最近的 $k$ 个邻居。

方法中，对于每个测试样本，循环处理其最近的 $k$ 个邻居，并根据其类别对当前测试样本的类别进行“投票”，最终排序选择被投票最多的类别，标记为当前测试样本的类别。

### 计算准确率

在上文中，我们已经投票决定了每个测试样本的类别，即已知每个测试样本的分类结果，这部分通过计算准确率对当前的 KNN 分类模型进行评价。

```python 
# Evaluate accuracy
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
```

本方法中的传入参数分别为：

- `testSet `：测试集，即所有的测试数据。
- `predictions `：上文的分类结果。

对于每个测试样本，对比其分类结果与真实类别是否一致，若一致则 `correct ` 加一，最终计算判断正确的测试样本占所有测试样本的百分比。

### 主函数

这部分将组织根据 `main` 方法的执行流程，梳理整个程序的思路。

```python
# main
def main():
    # load data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('./data/iris.data', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    # evaluate accuracy
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
```

在 `main` 方法中，主要处理流程如下：

1. 加载数据集，并制定数据的划分比例。在 `loadDataset` 方法中，根据指定的划分比例，随机划分数据集中的样本为测试集或训练集。
2. 对于每一个测试样本，通过计算向量间的欧式距离，找到与其最为相似的 $k$ 个训练样本。
3. 各个被选择的训练样本进行投票，选择最多票数的类别，标记为当前测试样本的类别。
4. 对比判定出的类别与数据集中真实的类别，并计算分类的准确率。

## 运行结果

### 准确率

项目最终的运行准确率为： $97.62$ %。

```shell
# 输入命令
$ python knn.pp
# 运行结果
Train set: 108
Test set: 42
Accuracy: 97.61904761904762%
```

### 运行结果分析

对于运行正确率的分析，进一步探究数据集的质量。

数据集的说明文件 `iris.names` 中有如下描述：

> The 35th sample should be: 4.9,3.1,1.5,0.2,"Iris-setosa"
>
> where the error is in the fourth feature.
>
> The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa"
>
> where the errors are in the second and third features.  

由于数据集本身只有 150 个样本，因此在进行模型分析时，数据集中的这两个错误值得考虑。
