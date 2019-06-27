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


# 2. Optimizing a neural network with backward propagation
# --------------------------------------------------------


# 3. Building deep learning models with keras
# ----------------------------------------------


# 4. Fine-tuning keras models
# ----------------------------------------------




