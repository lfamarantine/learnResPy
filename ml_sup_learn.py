# Supervised Learning with scikit-Learn
# -------------------------------------
# other popular ML libraries: TensorFlow, keras, scikit-learn integrates well with SciPy stack
# all ML models in scikit-learn as Python classes:
# - they implement the algorithms for learning & predicting
# - store the information learned from the data
# - .fit() & .predict() methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Classification
# -----------------

# ML: the art & science of: giving computers the ability to learn to make decisions from data
# without being explicitly programmed
# examples: learning to predict whether email is spam, clustering wikipedia entries into different categories

# types of ML:
# ---
# supervised: use labeled data
# unsupervised learning: uses unlabeled data
# reinforcement learning: software agents interact with an environment (automatically learn how t
# o optimize their behaviour given a system of rewards & punishments), draws inspiration from behavioural
# psychology (applications in economics, genetics, game playing), reinforcement learning was used in alphaGo

# supervised learning:
# - predictor variables/features and a target variable
# - 2 categories: classification (target variable consists of categories) or regression (target variable is continuous)

from sklearn import datasets

# k-Nearest Neighbors
# ---
# basic idea: predict the label of a data point by: 1. looking at the 'k' closest labeled data
# points & 2. taking majority vote
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('data/ml_house-votes-84.csv',sep=',')

# create arrays for the features and the response variable..
y = df['party'].values
X = df.drop('party', axis=1).values
# create a k-NN classifier with 6 neighbors..
knn = KNeighborsClassifier(n_neighbors=6)
# fit the classifier to the data..
knn.fit(X, y)
# predict the labels for the training data X..
y_pred = knn.predict(X)

# measuring model performance
# ---
# accuracy: fractions of correct predictions
# - which data should be used to predict accuracy?
# - how well will the model perform on unseen data?
# -> requirement for train/test-split, use train_test_split (75% train & 25% test is default)
from sklearn.model_selection import train_test_split

# split into training & test set..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42, stratify=y)
# create a k-NN classifier with 7 neighbors..
knn = KNeighborsClassifier(n_neighbors=7)
# fit the classifier to the training data..
knn.fit(X_train, y_train)
# accuracy..
knn.score(X_test, y_test)

# accuracy charts..
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# loop over different values of k..
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# plot..
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()



























