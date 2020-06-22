
# NAIVE BAYES ----------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
# source: https://pythonprogramminglanguage.com/python-machine-learning/

# create dataset
X = [[121, 80, 44], [180, 70, 43], [166, 60, 38], [153, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [174, 71, 40], [159, 52, 37], [171, 76, 42], [183, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
# create naive bayes classifier
gaunb = GaussianNB()
gaunb = gaunb.fit(X, Y)

# predict using classifier
prediction = gaunb.predict([[190, 70, 43]])
print(prediction)


# RANDOM FOREST --------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# source: https://pythonprogramminglanguage.com/python-machine-learning/

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X, y)


# SVM ------------------------------------------------------------------
from sklearn import svm
# source: https://pythonprogramminglanguage.com/python-machine-learning/

X = [[1, 1], [0, 0]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
# make predictions
print(clf.predict([[2., 2.]]))







