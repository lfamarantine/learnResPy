# Revision of Basic Statistics
# ----------------------------

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
from scipy.stats import bernoulli, binom


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

# normal distribution
# ---


# poission distribution
# ---
# represents a count / number of times something has happened
# parameter lambda
# example: in a 15-min time interval, there is a 20% probability that
# you will see at least one shooting star. What's the probability that
# you see at least one shooting star in the period of an hour?




