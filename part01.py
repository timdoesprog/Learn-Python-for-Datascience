# 25.09.2017
# https://www.youtube.com/watch?v=T5pRlIbr6gg
# Predict male or female based on arbitrary values


import numpy as np


X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])

y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male'])


def make_prediction(clf):
    prediction = clf.predict(np.array([[190, 70, 43]]))
    print(prediction)
    print('Mean Accuracy: ', clf.score(X, y))
    print()


# 1. Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
print('Decision Tree')
make_prediction(clf)


# 2. Gaussian Naive Bayes (GaussianNB)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, y)
print('Gaussian Naive Bayes')
make_prediction(clf)


# 3. K-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)
print('K-Nearest Neighbour')
make_prediction(clf)


# 4. Support Vector Classifier
from sklearn.svm import SVC

clf = SVC()
clf.fit(X, y)
print('Support Vector Classifier')
make_prediction(clf)
