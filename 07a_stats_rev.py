# asic Statistics in Py
# ----------------------
import pandas as pd

# 1. Probability & Sampling Distributions
# ---------------------------------------
# topics: conditional probabilities, baye's theorem, central limit theorem

# conditional probabilities
# ---
# Base case: You're testing for a disease and advertising that the test is 99% accurate; that is
# if you have the disease, you will test positive 99% of the time, and if you don't have the
# disease, you will test negative 99% of the time. Let's say that 1% of all people have the disease
# and someone tests positive. What's the probability that the person has the disease?
# .. prob = (0.99 * 0.01) / (0.99 * 0.01 + 0.01 * 0.99)


# Baye's theorem: p(a | b) = p(a & b) / p(b) = p(b | a) * p(a) / p(b)
# ---
# Baye's theorem applied: Out of the two coins, one is a real coin and the other one is a faulty coin
# with tails on both sides. You are blindfolded and forced to choose a random coin and then toss it in
# the air. The coin lands with tails facing upwards. Find the probability that this is the faulty coin.

# probabilities:
# prob(tails): 0.25
# prob(faulty): 0.75
# prob(tails & faulty): 0.5 * 1
# prob(faulty | tails):  0.5 / 0.75


# quick recap on list comprehensions
# ---
# conventional for loop..
x = [1,2,3,4]
out = []
for i in x:
    out.append(i**2)
print(out)

# list comprehension..
out = [i**2 for i in x]
print(out)


# law of large numbers
# ---
import numpy as np
from numpy.random import randint
# roll a dice 10 & 1000 times..
small = randint(1, 7, 10)
small_mean = np.mean(small)
print(small_mean)
large = randint(1, 7, 1000)
large_mean = np.mean(large)
print(large_mean)


# central limit theorem
# ---
import matplotlib.pyplot as plt

# a list of 1000 sample means of size 30..
means = [randint(1, 7, 30).mean() for i in range(1000)]
plt.hist(means)


# probability distributions
# ---
from scipy.stats import bernoulli, binom, norm


# bernoulli distribution
# ---
# .. 2 possible outcomes
# example: coin flip
plt.hist(bernoulli.rvs(p=0.5, size=1000))

# binomial distribution
# ---
# sum of outcomes of bernoulli trials
# binomial distribution is used to model the number of successful outcomes in trials
# where there is some consistent probability of success
plt.hist(binom.rvs(n=2, p=0.5, size=1000)) # inputs: k, p, n
# probability of 8 or less successes..
prob1 = binom.cdf(k=8, n=10, p=0.8)
print(prob1)
# probability of all 10 successes..
prob2 = binom.pmf(k=10, n=10, p=0.8)
print(prob2)

# normal distribution
# ---
data = norm.rvs(size=1000)
# true probability for greater than 2..
true_prob = 1 - norm.cdf(2)
print(true_prob)
# sample probability for greater than 2..
sample_prob = sum(obs > 2 for obs in data) / len(data)
print(sample_prob)

# poission distribution
# ---
# represents a count / number of times something has happened
# parameter lambda
# example: in a 15-min time interval, there is a 20% probability that
# you will see at least one shooting star. What's the probability that
# you see at least one shooting star in the period of an hour?


# 2. Exploratory Data Analysis
# ----------------------------
# centrality measures: mean, mode, median
# variability measures: variance, stdev, range
# modality: determined by # peaks a distribution contains
# other measures: skewness, kurtosis

# stdev by hand..
import math
nums = [1, 2, 3, 4, 5]
mean = sum(nums)/len(nums)
variance = sum(pow(x - mean, 2) for x in nums) / len(nums)
std = math.sqrt(variance)
print(std)
help(pow)

# Compute and print the actual result from numpy
real_std = np.array(nums).std()
print(real_std)

# encoding techniques..
# ---
from sklearn import preprocessing
import seaborn as sns
laptops = pd.read_csv("data/laptops.csv", delimiter=';')
# create the encoder and print our encoded..
encoder = preprocessing.LabelEncoder()
new_vals = encoder.fit_transform(laptops["Company"])
print(new_vals)
# one-hot encode Company for laptops2..
laptops2 = pd.get_dummies(data=laptops, columns=["Company"])
print(laptops2)

# pairplot..
df = pd.read_csv('data/airquality.csv', delimiter=',')
sns.pairplot(df)


# 3. Statistical Experiments and Significance Testing
# ---------------------------------------------------
import statsmodels as sm

# confidence intervals
# ---
from statsmodels.stats.proportion import proportion_confint
# # of heads in 50 fair coin flips - compute 99% CI..
heads = 27
confidence_int = proportion_confint(heads, 50, 0.01)
heads = binom.rvs(50, 0.5, size=10)
for val in heads:
    confidence_interval = proportion_confint(val, 50, .10)
    print(confidence_interval)

# hypothesis testing..
# ---
from statsmodels.stats.proportion import proportions_ztest
df = pd.read_csv('data/ab_data.csv')
results = df[['group','converted']]
conv_rates = results.groupby("group").mean()
num_control = results[results["group"]=="control"]['converted'].sum()
total_control = len(results[results["group"]=="control"])
num_treat = results[results['group'] == 'treatment']['converted'].sum()
total_treat = len(results[results['group'] == 'treatment'])
count = np.array([num_treat, num_control])
nobs = np.array([total_treat, total_control])

# z-test..
stat, pval = proportions_ztest(count, nobs, alternative="larger")
print('{0:0.3f}'.format(pval))
# t-test..
from scipy.stats import ttest_ind
# tstat, pval = ttest_ind(asus, toshiba) # ..comparing 2 price series

# power analysis
# ---
# power and sample size (how to calculate required sample size)
# -> power analysis!
# elements: effect size, significance level, power, sample size all lead
# to a larger sample size
from statsmodels.stats.proportion import proportion_effectsize
std_effect = proportion_effectsize(0.2,0.25)
# derive the required sample size..
from statsmodels.stats.power import  zt_ind_solve_power
sample_size = zt_ind_solve_power(effect_size=std_effect, nobs1=None, alpha=0.05, power=0.95)
print(sample_size)
# relationship between power and sample size..
sample_sizes = np.array(range(5, 100))
effect_sizes = np.array([0.2, 0.5, 0.8])
# esults object for t-test analysis..
from statsmodels.stats.power import TTestIndPower
results = TTestIndPower()
# plot of power analysis..
results.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)


# multiple testing
# ---
# bonferroni correction to account for multiple testing: alpha / n (= # tests performed)
# running 60 distinct hypothesis tests. Compute the probability of a Type I error for 60
# hypothesis tests with a single-test 5% significance level..
# error rate for 10 tests with 5% significance..
error_rate = 1 - (.95**(10))
print(error_rate)
# p-value adjustments for multiple testing..
from statsmodels.sandbox.stats.multicomp import multipletests
pvals = [.01, .05, .10, .50, .99]
# create a list of the adjusted p-values..
p_adjusted = multipletests(pvals, alpha=0.05, method='bonferroni')


# 4. Regression Models
# --------------------
df = pd.read_csv("data/weather_data_austin_2010.csv")
weather = df[['Temperature','DewPoint']]
from sklearn.linear_model import LinearRegression
X = np.array(weather['DewPoint']).reshape(-1,1)
y = weather['Temperature']
# linear regression..
lm = LinearRegression()
lm.fit(X, y)
coef = lm.coef_
preds = lm.predict(X)
plt.scatter(X, y)
plt.plot(X, preds, color='red')

# logistic regression..
from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# clf.fit(X, y_train.values)

# evlauting models..
# ---
# evaluating models: R-squared, mean absolute error (MAE), mean squared error (MSE)
# confusion matrices: dependeing on the question at hand, precision or recall might be more
# appropriate to concentrate on to evaluate performance
# .. Which error metric would you recommend for a dataset? It there aren't too many outliers,
# mean squared error would be a good choice to go with (rather than MAE).
# r-squared..
r2 = lm.score(X, y)
print(r2)
# mse, mae..
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(preds, y)
print(mse)
mae = mean_absolute_error(preds, y)
print(mse)
# confusion matrix..
from sklearn.metrics import confusion_matrix
# matrix = confusion_matrix(y_test, preds)
# precision = precision_score(y_test, preds)
# recall = recall_score(y_test, preds)


# missing data and outliers..
# ---
# techniques: dropping entire row or imputation
# imputation: constant value, randomly selected record, mead/meadian/mode, value estimated by another model
# useful functions: dropna, fillna, isnull


# bias-variance tradeoff..
# ---
# 3 types of errors: bias, variance, irreducable









