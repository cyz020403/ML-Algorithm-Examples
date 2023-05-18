import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class my_SVM:
    def __init__(self, eta=0.001, epoch=1000, random_state=42):
        self.eta = eta
        self.epoch = epoch
        self.random_state = random_state

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

    def predict(self, X):
        hyperplane = np.dot(X, self.w) + self.b
        result = np.where(hyperplane > 0, 1, -1)
        return result


iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

sc = StandardScaler()
X_std = sc.fit_transform(X)
y = np.where(y==0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, stratify=y)
mysvm = my_SVM()
mysvm.fit(X_train, y_train)
y_pred = mysvm.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy: ", accuracy)