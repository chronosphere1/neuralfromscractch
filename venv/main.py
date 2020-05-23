import sys
import numpy as np
import random

np.random.seed(0)

# inputs capital X. 3 samples
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # set random weight, inputs * neurons of standard normal distribution
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # create array of biases, set to 0
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# number of inputs * number of neurons
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)



print(layer2.output)



