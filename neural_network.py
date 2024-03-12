import copy

import numpy as np

sigmoid = "sigmoid"
relu = "relu"
CE = "crossEntropy"

class NN_model():
    def __init__(self, layers, learning_rate, epochs, cost_func):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_func = cost_func
        self.parameters = {}
        self.grads = {}
        self.losses = []

        if cost_func == CE:
            self.loss_backward = cross_entropy_loss_backward
            self.cost_func = compute_cross_entropy

        self.initialize_parameters()

    def initialize_parameters(self):
        for i in range(1, len(self.layers)):
            self.parameters["W" + str(i)] = np.random.randn(self.layers[i][0], self.layers[i - 1][0]) * 0.01
            self.parameters["b" + str(i)] = np.zeros((self.layers[i][0], 1))

    def forward(self, X):
        return model_forward(X, self.parameters, self.layers)

    def backward(self, Y, AL, caches):
        return model_backward(AL, Y, caches, self.loss_backward, self.layers)

    def train(self, X, Y):

        for i in range(self.epochs):
            AL, caches = self.forward(X)
            grads = self.backward(Y, AL, caches)
            update_parameters(self.parameters, grads, self.learning_rate)
            if i % 50 == 0:
                self.losses.append([i, self.cost_func(AL, Y)])

##forward
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
    A = 1 / (1 + np.exp(-Z)), Z
    return A
def relu_activation_forward(Z):
    Z_copy = copy.deepcopy(Z)

    Z_copy[Z_copy <= 0] = 0
    return (Z_copy, Z)

##backward

def model_backward(A, Y, caches, loss_backward, layers):
    dA = loss_backward(A, Y)

    grads = {}

    for i in reversed(range(len(caches))):#in cahces input layer is not included but in layers is
        current_cache = caches[i]

        dW, db, dA_prev = activation_backward(dA, current_cache, layers[i + 1][1])

        grads["dA" + str(i)] = dA_prev
        grads["dW" + str(i + 1)] = dW
        grads["db" + str(i + 1)] = db
        dA = dA_prev
    return grads

def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache

    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return (dW, db, dA_prev)

def activation_backward(dA, cache, activation):
    linear_cache , activation_cache = cache

    if activation == sigmoid:
        dZ = relu_activation_backward(dA, activation_cache)
        dW, db, dA_prev = linear_backward(dZ, linear_cache)

    if activation == relu:
        dZ = relu_activation_backward(dA, activation_cache)
        dW, db, dA_prev = linear_backward(dZ, linear_cache)

    return dW, db, dA_prev

def sigmoid_activation_backward(dA, Z):
    dZ = dA * Z * (1 - Z)
    return dZ

def relu_activation_backward(dA, Z):
    dZ = np.zeros(Z.shape)
    dZ[Z >= 0] = dA[Z >= 0]
    return dZ

def cross_entropy_loss_backward(AL, Y):
    dAL = - (np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))

    return dAL

####

def update_parameters(parameters, grads, learning_rate):

    params = copy.deepcopy(parameters)

    L = int(len(params) / 2)

    for i in range(L):
        params["W" + str(i + 1)] -= learning_rate * grads["dW" + str(i + 1)]
        params["b" + str(i + 1)] -= learning_rate * grads["db" + str(i + 1)]

    return params

#####cost functions
def compute_cross_entropy(AL, Y):
    m = Y.shape[1]
    cost = -np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)), axis=1, keepdims=False) / m

    return cost