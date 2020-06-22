# Supervised Learning in Python II
# --------------------------------
import pandas as pd
import numpy as np


# 1 Classification and Regression Trees
# -------------------------------------
# classification-tree: sequence of if-else questions about individual features to infer labels
# - captures non-linear relationships between features & labels
# - don't require feature scaling (eg. standardization)
# decision region: region in the feature space where all instances are assigned to one class label
# decision boundary: surface separating different decision regions

# DT-Classifier..
# ---
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('data/ml_dt_wbc.csv')
df2 = df.loc[:,['diagnosis','radius_mean','concave points_mean']]
X = df2.iloc[:,1:3]
y = df2.iloc[:,0]
# create training and test sets..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, criterion='entropy', random_state=1)
# fit dt to the training set..
dt.fit(X_train, y_train)
# predict test set labels..
y_pred = dt.predict(X_test)
print(y_pred[0:5])
# predict test set labels..
y_pred = dt.predict(X_test)
# compute test set accuracy..
acc = accuracy_score(y_pred, y_test)
print("Test set accuracy: {:.2f}".format(acc))

# compare to logistic-regression classification..
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=1)
logreg.fit(X_train, y_train)
# define a list called clfs containing the two classifiers logreg and dt..
clfs = [logreg, dt]


# DT-Regression
# ---
# Decision tree: a data structure consisting of a hierarchy of nodes (node: question or prediction)
# 3 kind of nodes:
# - root: no parent node, question giving rise to 2 children nodes
# - internal node: 1 parent node, question giving rise to 2 children nodes
# - leaf: 1 parent node, no children nodes -> prediction
# - min_samples_leaf: at each node use at least x% of the data for fitting
# unconstrained tree-learning:
# nodes are grown recursively -> at each node, split data based on feature f & split-point sp to maximize IG(node) - if
# IG(node)=0, declare the node a leaf. IG-criterion can be entropy, gini index or sth else (most of the time gini &
# entropy yield similar results, gini is faster & is default)

# RMSE: measures, on average, how much the model's predictions differ from the actual labels

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
df = pd.read_csv('data/ml_dt_auto.csv')
df_region = pd.get_dummies(df)
df_region = pd.get_dummies(df,drop_first=True)
y = df_region['mpg']
X = df_region.drop('mpg', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# instantiate dt..
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)
# fit to training data & calculate RMSE..
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
rmse_dt = MSE(y_pred, y_test)**(1/2)
print(rmse_dt)


# 2 The Bias-Variance Tradeoff
# ----------------------------
# generalization error of f_hat: bias^2 + variance + irreducible error (tells you how different f_hat vs f)
# bias term: error term that tells you, on average, how  much f_hat != f
# variance term: tells you how much f_hat is inconsistent over different training sets
# .. as the complexity of f_hat increases, the bias term decreases while the variance term increases

# Estimating the generalization error: can't be done directly as f is unknown, 1 dataset only & noise is unpredictable
# .. solution: split the data to training & test sets
# 1. fit f_hat to training set
# 2. evaluate the error of f_hat on the unseen test set
# 3. generalization error of f_hat similar test set error of f_hat
# note: test set should not be touched until we are confident about f_hat performance
# to remedy overfitting: decrease model complexity (eg. decrease max depth, increase min samples per leaf, gather
# more data)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.26, random_state=1)
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
print('CV RMSE: {:.2f}'.format((MSE_CV_scores.mean())**(1/2)))
# ..note that since cross_val_score has only the option of evaluating the negative MSEs, its output should be
# multiplied by negative one to obtain the MSEs


# ensamble learning
# ----
# limitations of CARTs:
# classification: can only produce orthogonal decision boundaries
# sensitive to small variations in the training set
# high variance: unconstrained CARTs may overfit the training set
# .. solution: ensemble learning!
# Ensemble learning: Train different models on the same dataset, let each model make its predictions & meta-model
# aggregates predictions of individual models. Final prediction is more robust & less prone to errors. Best results are
# obtained when models are skillful in different ways.
from sklearn.neighbors import KNeighborsClassifier as KNN
df = pd.read_csv('data/ml_dt_indian_liver_patient_preprocessed.csv')
y = df.loc[:,'Liver_disease']
X = df.drop('Liver_disease', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

SEED=1
# instantiate lr, knn & dt..
lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
# define the list classifiers..
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
# evaluate individual classifiers..
for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(X_train, y_train)
    # predict y_pred
    y_pred = clf.predict(X_test)
    # calculate accuracy
    accuracy = accuracy_score(y_pred, y_test)
    # evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

# voting classifier..
from sklearn.ensemble import VotingClassifier
# instantiate a VotingClassifier & fit..
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
# evaluate the test set predictions..
y_pred = vc.predict(X_test)
# calculate accuracy score..
accuracy = accuracy_score(y_pred, y_test)
print('Voting Classifier: {:.3f}'.format(accuracy))


# 3 Bagging and Random Forests
# ----------------------------
# Bagging is an ensemble method involving training the same algorithm many times using different subsets sampled
# from the training data.
# - 1 algorithm, different subsets of the training set
# - bagging stands for bootstrap aggregation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(random_state=1)
# instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_pred, y_test)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))
# ..  single tree dt would have achieved an accuracy of 63% which is 8% lower than bc's accuracy

# Out of Bag (OOB) Evaluation
# ---
# in bagging, some instances may be sampled several times for each model & other instances may not be sampled at all.
# On average, for each model, 63% of the training instances are sampled & the remaining 37% constitute the OOB
# instances. These OOB instances can be used for model validation instead of cross-validation.
dt = DecisionTreeClassifier(min_samples_leaf=8, random_state=1)
# instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, oob_score=True, random_state=1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_pred, y_test)
# evaluate OOB accuracy..
acc_oob = bc.oob_score_
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))


# Random Forest
# ---
# Each estimator is trained on a different bootstrap sample having the same size as the training set. RF introduces
# further randomization in the training of individual trees. d-features are sampled at each node without replacement
# (d < total # features).
# - feature importance: how much the tree nodes use a particular feature (weighted average) to reduce impurity
from sklearn.ensemble import RandomForestRegressor
# instantiate rf..
rf = RandomForestRegressor(n_estimators=25, random_state=2)
# fit rf to the training set..
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_test = MSE(y_test,y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
# ..test set RMSE achieved by rf is significantly smaller than that achieved by a single CART

# visualizing feature importance..
# ---
import matplotlib.pyplot as plt
dfcols = df.drop('Liver_disease', axis=1).columns
importances = pd.Series(data=rf.feature_importances_, index=dfcols)
# sort importance..
importances_sorted = importances.sort_values()
# draw a horizontal barplot..
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')


# 4 Boosting
# ----------
# Boosting refers to an ensemble method in which several models are trained sequentially with each model learning
# from the errors of its predecessors. Here, 2 types: 1) AdaBoost, 2) Gradient Boosting 3) Stochastic GB
# - boosting: ensemble method combining several weak learners to form a strong learner
# - weak learner: model doing slightly better than random guessing
# .. in boosting, train an ensemble of predictors sequentially.. each predictor tries to correct its predecessor


# AdaBoost
# ---
# adaboost (adaptive boosting): each predictor pays more attention to the instances wrongly predicted by its
# predecessor which is achieved by changing the weights of training instances. Each predictor is assigned a
# coefficient alpha which depends on the predictor's training error.
# - learning rate eta (0 < eta <= 1):
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
# instantiate ada..
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)
ada.fit(X_train, y_train)
y_pred_proba = ada.predict_proba(X_test)[:,1]
ada_roc_auc = roc_auc_score(y_test,y_pred_proba)
print('ROC AUC score: {:.2f}'.format(ada_roc_auc))


# Gradient Boosting
# ---
# Each predictor pays more attention to the instances wrongly predicted by its predecessor. In contrast to adaboost,
# weights of training instances are NOT changed. Instead, each predictor is trained using its predecessor's residual
# errors as labels.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
df = pd.read_csv('data/ml_dt_bikes.csv')
X = df.drop('cnt',axis=1).values
y = df.loc[:,'cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# instantiate gb..
gb = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
# evaluate..
rmse_test = MSE(y_test,y_pred)**(1/2)
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))


# Stochastic Gradient Boosting
# ---
# GB involves an exhaustive search procedure, each CART is trained to fit the best split points & features. This may
# lead to CARTs that use the same split points & features. To mitigate, one can use SGB:
# 1. each tree is trained on a random subset of rows of the training data
# 2. the sampled instances (40-80% of training set) are sampled without replacement
# 3. features are sampled without replacement when chosing split points
# .. this results in further ensemble diversity & adds further variance to the ensemble of trees
from sklearn.ensemble import GradientBoostingRegressor
# instantiate sgbr..
sgbr = GradientBoostingRegressor(max_depth=4, subsample=0.9, max_features=0.75, n_estimators=200, random_state=2)
sgbr.fit(X_train,y_train)
y_pred = sgbr.predict(X_test)
# evaluate..
rmse_test = MSE(y_test,y_pred)**(1/2)
print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))


# 5 Model Tuning
# --------------
# The hyperparameters of a machine learning model are parameters that are not learned from data.
# - search for optimal hyperparameters for learning algorithm that yields an optimal score
# - optimal score default in sklearn is accuracy (classification) & r-squared (regression)
# - cross-validation is generalization performance measurement

# approaches to hyperparameter tuning:
# 1. grid search
# 2. random search
# 3. bayesian optimization
# 4. genetic algorithms

# hyperparameter tuning is computationally expensive & sometimes leads to very slight improvement -> weight the
# impact of tuning on the project
# view rf-hyperparameters: rf.get_params()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
df = pd.read_csv('data/ml_dt_indian_liver_patient_preprocessed.csv')
y = df.loc[:,'Liver_disease']
X = df.drop('Liver_disease', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CART tuning..
# ---
# set tree's parameter grid..
params_dt = {'max_depth':[2,3,4],'min_samples_leaf':[0.12,0.14,0.16,0.18]}
# search for optimal tree..
dt = DecisionTreeClassifier(random_state=1)
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)
# evaluate optimal tree..
grid_dt.fit(X_train, y_train)
# extract best estimator & predict / evaluate..
best_model = grid_dt.best_estimator_
y_pred_proba = grid_dt.predict_proba(X_test)[:,1]
test_roc_auc = roc_auc_score(y_test,y_pred_proba)
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

# RF tuning..
# ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
df = pd.read_csv('data/ml_dt_bikes.csv')
X = df.drop('cnt',axis=1).values
y = df.loc[:,'cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params_rf = {'n_estimators':[100,350,500],'max_features':['log2','auto','sqrt'], 'min_samples_leaf':[2,10,30]}
# search for optimal tree..
rf = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=2)
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
# train the tree..
grid_rf.fit(X_train, y_train)
# extract the best estimator, predict & evaluate..
best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = MSE(y_test,y_pred)**(1/2)
print('Test RMSE of best model: {:.3f}'.format(rmse_test))


