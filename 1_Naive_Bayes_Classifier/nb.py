import pandas as pd
import math

# 加载数据集
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', \
                    header=None)

# 将数据集分为训练集和测试集
train_data = data.sample(frac=0.7, random_state=1)
test_data = data.drop(train_data.index)

# 将训练数据按类别分组
groups = train_data.groupby(train_data.iloc[:, -1])

# 计算每个类别的先验概率和每个特征的条件概率
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

# 定义朴素贝叶斯分类器
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

# 对测试集进行分类并计算准确率
correct = 0
for i in range(len(test_data)):
    sample = test_data.iloc[i, :-1]
    true_class = test_data.iloc[i, -1]
    predicted_class = predict_class(sample, priors, conditionals)
    if predicted_class == true_class:
        correct += 1
accuracy = correct / len(test_data)

# 显示准确率
print('准确率为：', accuracy)
