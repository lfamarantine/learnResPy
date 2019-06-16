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

help(pd)


# 2. Regression
# -------------
# cross-validation: more folds -> more computationally expensive
# ..maximizes the amount of data that is used to train the model, as during the course of training, the model
# is not only trained, but also tested on all of the available data.

# timing cv-procedure: cross_val_score(reg, X, y, cv=..)

# Why regularize? Linear regression minimizes a loss-function, it chooses a coefficient for each variable. Large
# coefficients can lead to over-fitting. In practice, one can alter the loss-function so that it penalizes large
# coefficients, which is called regularization!

# Types of regularized regressions: Ridge & Lasso
# 1. Ridge regression: models are penalized for coefficients with a large multitude (pos./neg.)
# - alpha-parameter needs to be chosen (similar to choosing k in kNN) -> this is hyperparameter tuning!
# - alpha controls model complexity (alpha=0: standard OLS) -> small alpha: can lead to overfitting,
# large alpha: can lead to underfitting

# 2. Lasso regression: can be used to select important features of a dataset because it shrinks the coefficients
# of less important features to exactly 0 (lasso's power is to determine the important features)


# 3. Fine-tuning your model
# -------------------------
# class-imbalance email example: imagine in spam classification 99% of emails are real, 1% spam. Assume the classifier
# on this sample to determine spams is 99% accurate. However, it's not good at actually predicting spam and it fails
# at its original purpose. Need for more nuanced metrics!!

# confusion matrix metrics:
# i.   accuracy: (tp + tn) / (tp + tn + fp + fn)
# ii.  precision: tp / (tp + fp) (also called positive predictive value - PPV)
# iii. recall: tp / (tp + fn) (also called sensitivity, hit-rate, true-positive rate)
# iv.  F1 score: 2 * (precision * recall) / (precision + recall) -> harmonic mean of precision & recall
# .. high precision: not many real emails predicted as spam
# .. high recall: predicted most spam emails correctly













