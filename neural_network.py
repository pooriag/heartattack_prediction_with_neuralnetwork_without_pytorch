import numpy as np

sigmoid = "sigmoid"
relu = "relu"

class NN_model():
    def __init__(self, layers, learning_rate, epochs, cost_func):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_func = cost_func
        self.parameters = {}
        self.grads = {}

        self.initialize()

    def initialize(self):
        for i in range(1, len(self.layers)):
            self.parameters["W" + str(i)] = np.random.randn(self.layers[i][0], self.layers[i - 1][0]) * 0.01
            self.parameters["b" + str(i)] = np.zeros(self.layers[i][0], 1)

    def forward(self, X):
        model_forward(X, self.parameters, self.layers)



def model_forward(A0, parameters, layers):

    caches = []
    A = A0

    for i in range(1, len(layers)):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], layers[i][1])
        caches.append(cache) ##indexin starts with 0 for the first hidden layer

    return A, caches

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache

def activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == sigmoid:
        A, activation_cache = sigmoid_activation_forward(Z)
        cache = (linear_cache, activation_cache)

    if activation == relu:
        A, activation_cache = relu_activation_forward(Z)
        cache = (linear_cache, activation_cache)

    return A, cache

def sigmoid_activation_forward(Z):
    return 1 / 1 + np.exp(-Z), Z

def relu_activation_forward(Z):
    return max(0, Z), Z