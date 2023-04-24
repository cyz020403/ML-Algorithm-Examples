import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

# Define the Node class for the decision tree
class Node:
    def __init__(self, attribute=None, threshold=None, label=None):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        
# Define the ID3 decision tree algorithm
def ID3(X, y, depth=0, max_depth=5):
    if depth == max_depth or len(np.unique(y)) == 1:
        label = np.bincount(y).argmax()
        return Node(label=label)
    
    num_features = X.shape[1]
    best_attribute = None
    best_threshold = None
    best_gain = -math.inf
    
    for i in range(num_features):
        feature_values = np.unique(X[:, i])
        for j in range(1, len(feature_values)):
            threshold = (feature_values[j] + feature_values[j-1]) / 2
            left_idx = np.where(X[:, i] <= threshold)
            right_idx = np.where(X[:, i] > threshold)
            left_y = y[left_idx]
            right_y = y[right_idx]
            if len(left_y) == 0 or len(right_y) == 0:
                continue
            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_attribute = i
                best_threshold = threshold
                best_gain = gain
                
    if best_gain == -math.inf:
        label = np.bincount(y).argmax()
        return Node(label=label)
    
    node = Node(attribute=best_attribute, threshold=best_threshold)
    left_idx = np.where(X[:, best_attribute] <= best_threshold)
    right_idx = np.where(X[:, best_attribute] > best_threshold)
    left_X, left_y = X[left_idx], y[left_idx]
    right_X, right_y = X[right_idx], y[right_idx]
    node.left = ID3(left_X, left_y, depth+1, max_depth)
    node.right = ID3(right_X, right_y, depth+1, max_depth)
    return node

# Define the entropy and information gain functions
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = np.sum(probabilities * -np.log2(probabilities))
    return entropy

def information_gain(parent, left, right):
    parent_entropy = entropy(parent)
    left_entropy = entropy(left)
    right_entropy = entropy(right)
    left_weight = len(left) / len(parent)
    right_weight = len(right) / len(parent)
    gain = parent_entropy - left_weight * left_entropy - right_weight * right_entropy
    return gain

# Train the decision tree
tree = ID3(X_train, y_train, max_depth=5)

# Define the predict function
def predict(X, tree):
    if tree.label is not None:
        return tree.label
    if X[tree.attribute] <= tree.threshold:
        return predict(X, tree.left)
    else:
        return predict(X, tree.right)
    
# Make predictions
y_pred = np.array([predict(x, tree) for x in X_test])

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the decision tree with matplotlib
def plot_tree(tree, X, y, spacing="    "):
    if tree.label is not None:
        print(spacing + "Predict", tree.label)
        return
    # three decimal places
    threshold = round(tree.threshold, 3)
    print(spacing + str(tree.attribute) + " <= " + str(threshold))
    print(spacing + "T->")
    plot_tree(tree.left, X, y, spacing + "  ")
    print(spacing + "F->")
    plot_tree(tree.right, X, y, spacing + "  ")

plot_tree(tree, X, y)