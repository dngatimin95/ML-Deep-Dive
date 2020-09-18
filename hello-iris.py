from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_dataset = load_iris()
x = iris_dataset.data
y = iris_dataset.target
print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])
print(iris_dataset['data'].shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size = 0.4)

tree = DecisionTreeClassifier(max_depth = 3, random_state = 1)
tree.fit(x_train, y_train)
pred_tree = tree.predict(x_test)
print("Accuracy of Decision Tree: " + str(accuracy_score(pred_tree, y_test)))
print(tree.feature_importances_)

plt.figure(figsize = (10,8))
plot_tree(tree, feature_names = iris_dataset['feature_names'], class_names = iris_dataset['target_names'], filled = True)
plt.show()

gaus = GaussianNB()
gaus.fit(x_train, y_train)
pred_gaus = gaus.predict(x_test)
print("Accuracy of Gaussian: " + str(accuracy_score(y_test,pred_gaus)))

logres = LogisticRegression()
logres.fit(x_train, y_train)
pred_log = logres.predict(x_test)
print("Accuracy of Logistic Regression: " + str(accuracy_score(y_test,pred_log)))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
print("Accuracy of KNN: " + str(accuracy_score(y_test,pred_knn)))

x_new = np.array([[4, 1.2, 2, 0.5]])
prediction = knn.predict(x_new)
print("Predicted target name: " + str(iris_dataset['target_names'][prediction][0]))
