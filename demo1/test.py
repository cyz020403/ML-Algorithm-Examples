import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.read_csv("iris.data", header=None, names=["sepal_length", "sepal_width", 
#                                                     "petal_length", "petal_width", "class"])

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
                 , header=None)
# print(df)

# 将数据集分为训练集和测试集
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
