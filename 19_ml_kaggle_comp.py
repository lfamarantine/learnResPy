# Kaggle Competition Case Study
# -----------------------------
# the challenge: Budgets for schools are huge, complex & not standardized. Hundreds of hours
# are spent each year manually labelling them.
# the goal: build a ml algorithm to automate this process
# example line item is; "Algebra books for 8th grade students" with labels: math,
#                        textbooks, middle-school (target variable)
# supervised classification problem!
import pandas as pd
import numpy as np

# 1. Exploring the raw data
# -------------------------
# review of lambda functions (1-line functions): square = lambda x: x*x -> square(2)
# log loss: provides a steep penalty for predictions that are both wrong and confident
# df.dtypes.value_counts()
def compute_log_loss(predicted, actual, eps=1e-14):
    """Computes the logartithmic loss between predicted & actual when these are 1D arrays.
    :param predicted: The predicted probabilities as floats between 0-1
    :param actual: The actual binary labels. Either 0 or 1.
    :param eps (optional): log(0) is inf, so we need to offset our predicted values by eps from 0 or 1.
    """
    predicted = np.clip(predicted, eps, 1-eps)
    loss = -1 * np.mean(actual*np.log(predicted)) + (1-actual) * np.log(1-predicted)
    return loss


# 2. Creating a simple first model
# --------------------------------
# - always a good approach to start with a very simple model
# - gives a sense of how challenging the problem is
# - many more things can go wrong in complex models
# - how much signal can we pull out using basic methods?

# NLP
# ---
# data can be: text, speech, documents
# tokenization: splitting strings into segments
# .. tokenize on whitespace -> petro-vend fuel and fluids: petro-vend | fuel | and | fluids
# alternative tokenize methods: whitespace & punctuation simultanously
# in scikit-learn: CountVectorizer()
#  1. tokenizes all the strings
#  2. builds a vocabulary (notes down all the words that appear)
#  3. counts the occurences of each token in the vocabulary


# 3. Improving your model
# -----------------------


# 4. Learning from the experts
# ----------------------------



