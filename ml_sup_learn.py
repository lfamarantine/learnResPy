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


# reshaping data before modelling..
df = pd.read_csv('data/ml_gm_2008_region.csv')
y = df['life'].values
X = df['fertility'].values
# ..reshape:
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# linear regression
# ---
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# the prediction space..
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)
# fit the model to the data..
reg.fit(X, y)
# compute predictions over the prediction space..
y_pred = reg.predict(prediction_space)
# R-squared..
print(reg.score(X, y))

# train/test splitting..
# ---
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# create training and test sets..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
# R^2 and RMSE..
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# cross-validation..
# ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
# compute 5-fold cross-validation scores..
cv_scores = cross_val_score(reg, X, y, cv=5)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# regularization: lasso, ridge & elastic net
# ---
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.4, normalize=True)
lasso.fit(X,y)
print(lasso.coef_)

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.4, normalize=True)
ridge.fit(X,y)
print(ridge.coef_)

# elastic net..
# ---
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)
# create the hyperparameter grid..
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
# instantiate the ElasticNet regressor..
elastic_net = ElasticNet()
# setup the GridSearchCV object..
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
# fit it to the training data..
gm_cv.fit(X_train,y_train)
# predict on the test set and compute metrics..
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)


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

# confusion matrix..
# ---
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('data/ml_diabetes.csv')
y = df['diabetes'].values.reshape(-1,1)
X = df.drop('diabetes', axis=1).values
# create training and test set..
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)
# instantiate a k-NN classifier..
knn = KNeighborsClassifier(n_neighbors=6)
# fit the classifier to the training data..
knn.fit(X_train, y_train)
# predict the labels of the test data..
y_pred = knn.predict(X_test)
# generate the confusion matrix and classification report..
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# logistic regression
# ---
# logistic regression outputs probabilities with p > / < 0.5 -> 1 / 0
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
logreg = LogisticRegression()
# fit model..
logreg.fit(X_train, y_train)
# predict the labels of the test set..
y_pred = logreg.predict(X_test)
# model statistics..
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)


# visual model performance evaluation
# ---
# - roc-curve
# - area under the curve (the larger the better the model)
# - precision-recall curve

# ROC curve..
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# AUC computation..
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# simple AUC score..
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# cross-validated AUC scores..
cv_auc = cross_val_score(logreg, X, y,cv=5,scoring='roc_auc')
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


# hyperparameter-tuning
# ---
# how to choose hyperparameter?
# - try a bunch of different hyperparameter values
# - fit all of them separately & see how it performs - choose best performing one
# - essential to use cross-validation

# grid-search & randomized-search..
# grid-search can be computationally expensive, use randomized-search where a fixed number of hyperparameter settings

# 1 grid-search..
# is sampled from specified probability distributions
from sklearn.model_selection import GridSearchCV
# setup the hyperparameter grid..
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
# instantiate a logistic regression classifier..
logreg = LogisticRegression()
# instantiate the GridSearchCV object..
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# fit it to the data..
logreg_cv.fit(X,y)
# tuned parameters and score..
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# 2 randomized-search..
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
# setup the parameters and distributions to sample from..
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
# instantiate a Decision Tree classifier..
tree = DecisionTreeClassifier()
# instantiate the RandomizedSearchCV object..
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
# fit it to the data..
tree_cv.fit(X,y)
# tuned parameters and score..
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# hold-out set for final validation
# ---
# how well does the model perform on never seen data?
# - using all data for cv isn't ideal
# - split data into training and hold-out set at beginning & perform grid-search cv on training data
# - choose best hyper-parameters & evaluate on hold-out set


# 4. Pre-processing and pipelines
# ------------------------------
# - scikit-learn won't accept categorical features by default -> decode to dummy variables
# - dealing with categorical features: sckit-learn: OneHotEncoder() & pandas: get_dummies()
df = pd.read_csv('data/ml_gm_2008_region.csv')
# create a boxplot of life expectancy per region..
df.boxplot('life', 'Region', rot=60)

# categorical variables..
# ---
# create dummy variables..
df_region = pd.get_dummies(df)
# always n-1 dummy variables given n: # categorical variables - so drop unneeded dummy variable "Region_America"
df_region = pd.get_dummies(df,drop_first=True)
y = df_region['life']
X = df_region.drop('life', axis=1).values

# estimate a ml-model..
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
ridge = Ridge(alpha=0.5, normalize=True)
ridge_cv = cross_val_score(ridge,X,y,cv=5)
print(ridge_cv)

# handling missing data..
# ---
df = pd.read_csv('data/ml_house-votes-84.csv')
df[df == '?'] = np.nan
# number of nan's..
print(df.isnull().sum())
# drop missing values..
df = df.dropna()

# imputation..
# ---
# 1 indirect..
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean',axis=0)
imp.fit(X)
X = imp.transform(X)

# pipeline objects..
# ---
# impute the dataset, split data & estimate model..
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# prepare dataset..
df = pd.read_csv('data/ml_house-votes-84.csv')
df[df == '?'] = np.nan
df[df == 'y'] = 1
df[df == 'n'] = 0
y = df['party']
X = df.drop('party', axis=1).values

# setup the pipeline with the required steps..
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)), ('SVM', SVC())]
# create the pipeline..
pipeline = Pipeline(steps)
# create training and test sets..
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
# fit the pipeline to the train set..
pipeline.fit(X_train,y_train)
# predict the labels of the test set..
y_pred = pipeline.predict(X_test)
# compute metrics..
print(classification_report(y_test, y_pred))


# scaling data..
# ---
# - many models use some form of distance to inform them
# - features on larger scales can unduly influence the model -> k-NN uses distance explicitly when making predictions
# - requirement for features to be on similar scale
from sklearn.preprocessing import scale
df = pd.read_csv('data/ml_gm_2008_region.csv')
X = df.drop(['fertility','Region'], axis=1).values
y = df['Region']
# scale the features..
X_scaled = scale(X)

# scaling in a pipeline..
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# aetup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
# create the pipeline..
pipeline = Pipeline(steps)
# create train and test sets..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# fit the pipeline to the training set..
knn_scaled = pipeline.fit(X_train, y_train)
# instantiate and fit a k-NN classifier to the unscaled data..
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
# metrics..
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))


# pipelines extended: classification & regression..
# ---

# 1 classification..
df = pd.read_csv('data/ml_diabetes.csv')
y = df['diabetes'].values.reshape(-1,1)
X = df.drop('diabetes', axis=1).values

# pipeline setup..
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
# specify the hyperparameter space..
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
# create train and test sets..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=21)
# instantiate the GridSearchCV object..
cv = GridSearchCV(pipeline, parameters, cv=3)
# fit to the training set..
cv.fit(X_train,y_train)
# predict the labels of the test set..
y_pred = cv.predict(X_test)
# compute and print metrics..
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

# 2 regression..
df = pd.read_csv('data/ml_gm_2008_region.csv')
y = df['life'].values
X = df.drop(['life','Region'],axis=1).values
# setup the pipeline steps & create the pipeline..
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]
pipeline = Pipeline(steps)
# specify the hyperparameter space..
parameters = {'elasticnet__l1_ratio': np.linspace(0, 1, 30)}
# create train and test sets..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# create the GridSearchCV object..
gm_cv = GridSearchCV(pipeline, parameters)
# fit to the training set..
gm_cv.fit(X_train, y_train)
# metrics..
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))







