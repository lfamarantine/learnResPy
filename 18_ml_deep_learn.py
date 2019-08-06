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

# weight changes effect on accuracy showcase - 1 data-point
# ---

# data point you will make a prediction for..
input_data = np.array([0, 3])
# sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]}
# the actual target value (used to calculate the error)..
target_actual = 3
# make prediction using original weights..
model_output_0 = predict_with_network(input_data, weights_0)
# calculate error..
error_0 = model_output_0 - target_actual
# create weights that cause the network to make perfect prediction (3)..
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]}
# make prediction using new weights..
model_output_1 = predict_with_network(input_data,weights_1)
# calculate error..
error_1 = model_output_1 - target_actual
print(error_0)
print(error_1)

# weight changes effect on accuracy showcase - multiple data-points
# ---
from sklearn.metrics import mean_squared_error
input_data = [np.array([0,3]),np.array([1,2]),np.array([-1,-2]),np.array([4,0])]
model_output_0 = []
model_output_1 = []
# loop over input_data..
for row in input_data:
    # append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    # append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# calculate the mean squared error..
target_actuals = np.array([1,3,5,7])
mse_0 = mean_squared_error(target_actuals, model_output_0)
mse_1 = mean_squared_error(target_actuals, model_output_1)
print("Mean squared error with weights_0: %f" % mse_0)
print("Mean squared error with weights_1: %f" % mse_1)


# calculating slopes & updating weights..
# ---

# 1 update cycle..
weights = np.array([0,2,1])
input_data = np.array([1,2,3])
target = 0
# set the learning rate..
learning_rate = 0.01
# calculate the predictions..
preds = (weights * input_data).sum()
# calculate the error..
error = preds - target
# calculate the slope..
slope = 2 * input_data * error
# update the weights..
weights_updated = weights - learning_rate*slope
# get updated predictions..
preds_updated = (weights_updated * input_data).sum()
# calculate updated error..
error_updated = preds_updated-target
# original error & updated error..
print(error)
print(error_updated)


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
# creating a good model requires experimentation!

# understanding model optimization
# difficulties:
# - simultaneously optimising 1000s of parameters with complex relationships
# - updates may not improve model meaningfully (updates to small due to low learning rate or
#   too large if learning rate is too high)
# - dead neuron problem: due to slope being zero, all estimations result in zero -> alternative
#   activation function might help (poor choice of activation function)
# .. this points can prevent the model from showing improved loss in first few epochs

# validation is deep learning:
# - commonly use a validation split rather than cross-validation
# - deep learning widely used on big data & hence single validation score based on large amount of data is reliable
# - repeated training from cross-validation would take a long time

# early stopping: keep training while validation score is improving & stop when not
# .. patience parameter is set to say how many epochs the model can go and improve
# before we stop training (2 or 3 are usually reasonable values)
# epochs: because optimization will automatically stop when it is no longer helpful, it is okay to
# specify the maximum number of epochs as 30 for example rather than using the default of 10

# model capacity
# --
# - closely related to overfitting / underfitting
# - adding more layers & nodes increases capacity
# .. start with small network & keep increasing capacity until validation score is no longer improving


# evaluating model accuracy with validation split & optimizing optimization
# ---
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
df = pd.read_csv('data/ml_dl_titanic_all_numeric.csv')
predictors = df.drop('survived', axis=1).values
# convert the target to categorical (one hot encoding)..
target = to_categorical(df.survived)
# preparation..
n_cols = predictors.shape[1]
input_shape = (n_cols,)
# specify the model..
model = Sequential()
# add 2 layers & output layer..
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
# compile the model..
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# define early_stopping_monitor..
early_stopping_monitor = EarlyStopping(patience=2)
# fit the model
model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks=[early_stopping_monitor])
# note: verbose=False in the fitting commands to print out fewer updates!




