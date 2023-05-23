from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the iris dataset from a local file or online
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(url, header=None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = iris.drop_duplicates()

# Convert the class labels to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(iris['class']))}
# print(class_mapping)
iris['class'] = iris['class'].map(class_mapping)

# Shuffle and split
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# conda activate cyz01
# cd C:\Users\17875\Desktop\ML-Algorithm-Examples\4_SVM
