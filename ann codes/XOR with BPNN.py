#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Set the random seed for reproducibility
np.random.seed(1)

# Define the input and output data for the XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the hyperparameters for the network
epochs = 10000
learning_rate = 0.1
hidden_layer_size = 25

# Initialize the weights for the input layer and the hidden layer
input_layer_weights = np.random.uniform(size=(2, hidden_layer_size))
hidden_layer_weights = np.random.uniform(size=(hidden_layer_size, 1))

# Train the neural network using backpropagation
for i in range(epochs):

    # Forward propagation
    hidden_layer_output = sigmoid(np.dot(X, input_layer_weights))
    output_layer_output = sigmoid(np.dot(hidden_layer_output, hidden_layer_weights))

    # Backpropagation
    output_layer_error = y - output_layer_output
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = output_layer_delta.dot(hidden_layer_weights.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Update the weights
    hidden_layer_weights += hidden_layer_output.T.dot(output_layer_delta) * learning_rate
    input_layer_weights += X.T.dot(hidden_layer_delta) * learning_rate

# Test the neural network
hidden_layer_output = sigmoid(np.dot(X, input_layer_weights))
output_layer_output = sigmoid(np.dot(hidden_layer_output, hidden_layer_weights))
print(output_layer_output)


# In[ ]:




