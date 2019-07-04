# Deep Learning in Python
# -----------------------
# deep learning libraries: 1) keras & Tensorflow
# foundations for advanced applications to image, text, audio, video & more
# - neural networks account for interactions really well
# - deep learning uses especially powerful for neural networks

import pandas as pd
import numpy as np

# 1. Basics of deep learning and neural networks
# ----------------------------------------------
# forward propagation: how neural networks use data to make predictions

# coding the simplest forward propagation..
# ---
# -> for reference, see meta/ml_learn_dl_froward_propagation2.PNG for the following:
input_data = np.array([3,5])
weights = {'node_0':np.array([2,4]), 'node_1':np.array([4,-5]), 'output':np.array([2,7])}
# calculate node 0 value..
node_0_value = (input_data * weights['node_0']).sum()
# calculate node 1 value..
node_1_value = (input_data * weights['node_1']).sum()
# put node values into array..
hidden_layer_outputs = np.array([node_0_value, node_1_value])
# calculate output..
output = (hidden_layer_outputs * weights['output']).sum()
print(output)

# activation functions..
# ---
# the multiply-sum-process in the network is only half the story..
# to maximize its predictive power of neural networks, activation functions are used in the hidden layer at each node!
# activation function: allows the model to capture non-linearities
# - applied to node inputs to produce node output
# - standard activation function today in industry: ReLU (Rectified Linear Activation)

# an example relu-function..
def relu(input):
    '''Define your relu activation function here'''
    # calculate the value for the output of the relu function..
    output = max(input, 0)
    return (output)

# calculate node 0 value..
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)
# calculate node 1 value..
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)
# put node values into array..
hidden_layer_outputs = np.array([node_0_output, node_1_output])
# calculate model output..
model_output = (hidden_layer_outputs * weights['output']).sum()
print(model_output)


# applying the network to many observations/rows of data..
# ---
input_data = [np.array([3,5]), np.array([1,-1]), np.array([0,0]), np.array([8,4])]
weights = {'node_0':np.array([2,4]), 'node_1':np.array([4,-5]), 'output':np.array([2,7])}

def predict_with_network(input_data_row, weights):
    # calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    # calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    # put node values into array
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    # calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    # return model output
    return (model_output)


# create empty list to store prediction results..
results = []
for input_data_row in input_data:
    results.append(predict_with_network(input_data_row, weights))
print(results)

# difference between modern & historical deep learning models: model DL-algorithms use many hidden layers
# representation learning:
# - deep neural networks internally build representations of patterns in data
# - partially replace the need for feature engineering
# - DL also call representation learning because subsequent layers build increasingly sophisticated representations
#   of raw data
# - modeler doesn't need to specify the interactions
# - when you train the model, the neural network gets weights that find the relevant patterns to make predictions

# remarks:
# - last layers in a neural network capture more complex "higher-level" interactions
# - weights that determine the features/interactions are created through the model training process that sets them
#   to optimize predictive accuracy


# 2. Optimizing a neural network with backward propagation
# --------------------------------------------------------
# loss function: aggregates errors in predictions from many data points into a single number & measure of
# model's predictive performance
# - lower loss function value means a better model
# - goal: find the weights that give the lowest value for the loss function -> This is performed
#         with an algo called Gradient Descent

# calculating slopes & updating weights..
# ---


# backpropagation
# ---
# backpropagation process:
# 1. go back 1 layer at a time
# 2. gradients of weights are product of:
#       1) node value feeding into that weight
#       2) slope of loss function wrt. node it feeds into
#       3) slope of activation function at the node it feeds into
# - stochastic gradient descent: when slopes are calculated on 1 batch (subset of data) at a time


# 3. Building deep learning models with keras
# ----------------------------------------------
# model building steps:
# 1. specify architecture
# 2. compile
# 3. fit
# 4. predict
df = pd.read_csv('data/ml_dl_hourly_wages.csv')
target = np.array(df['wage_per_hour'])
predictors = np.array(df.drop('wage_per_hour', axis=1))

# 1 regression problems
# ---

# specifying a neural network model
# ---
import keras
from keras.layers import Dense
from keras.models import Sequential

# set up the model..
n_cols = predictors.shape[1]
model = Sequential()
# add the 1st layer..
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
# add the 2nd layer..
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
# add the output layer..
model.add(Dense(1))
# compile the model..
model.compile(optimizer='adam', loss='mean_squared_error')
# fit the model..
model.fit(predictors, target)

# compiling & fitting a model..
# ---
# compile method has 2 arguments to choose:
# 1. what optimizer to choose (controls the learning rate) - many options & mathematically complex
#  -> "Adam" is usually a good choice
# 2. loss function (MSE is most common choice for regression problems)

# fitting: applying backpropagation & gradient descent with data to update the weights
# - scaling data before fitting can ease optimization

# 2 classification problems
# ---
from keras.utils import to_categorical
# data prep..
df = pd.read_csv('data/ml_dl_titanic_all_numeric.csv')
predictors = df.drop('survived', axis=1).values
# convert the target to categorical (one hot encoding)..
target = to_categorical(df.survived)

# major differences compared to regression problems:
# - loss function: 'categorical_crossentropy' (most common lost function for classification) -> similar to
#   log loss: lower is better
# - metrics: add metrics='accuracy' to compile step for easy-to-understand diagnostics
# - output layer has separate node for each possible outcome & uses 'softmax' activation (sum up to 1 so that they
#   can be interpreted as probabilities)

# # predictive features..
n_cols = 10
# set up the model..
model = Sequential()
# add the 1st layer
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
# add the output layer
model.add(Dense(2, activation='softmax'))
# compile the model
# note: optimizer 'sgd': Stochastic Gradient Descent
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(predictors, target)

# saving, reloading & using a model
# ---


# 4. Fine-tuning keras models
# ----------------------------------------------




