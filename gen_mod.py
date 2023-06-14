import numpy as np

import utils
from activations import *


class GeneratorLayer:
    def __init__(self, prev_size, curr_size, activation_func, activation_func_der):
        self.weights = np.array(np.random.normal(0, 0.1, size=(curr_size, prev_size)))
        self.biases = np.array(np.random.normal(0, 0.01, size=curr_size))
        self.raw_activation = np.zeros(curr_size)
        self.activated = np.zeros(curr_size)

        self.weights_der = np.zeros((curr_size, prev_size))
        self.biases_der = None
        self.raw_activation_der = np.zeros(curr_size)
        self.activated_der = np.zeros(curr_size)

        self.activation_func = np.vectorize(activation_func)
        self.activation_func_der = np.vectorize(activation_func_der)

    def __str__(self):
        return str(self.weights) + "  " + str(self.biases)

    def forwardprop(self, prev_layer):
        self.raw_activation = np.dot(self.weights, prev_layer.activated) + self.biases
        self.activated = self.activation_func(self.raw_activation)

    def backprop(self, prev_layer, learning_rate):
        self.raw_activation_der = self.activation_func_der(self.raw_activation)
        self.biases_der = self.activated_der * self.raw_activation_der
        self.weights_der = np.outer(self.biases_der, prev_layer.activated)
        self.weights = self.weights - learning_rate * self.weights_der
        self.biases = self.biases - learning_rate * self.biases_der
        prev_layer.activated_der = np.dot(np.transpose(self.weights), self.biases_der)


class Generator:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate
        self.last_layer = 0
        for prev_size, curr_size in zip([1] + layer_sizes, layer_sizes):
            # if self.last_layer == len(layer_sizes) - 1:
            #     self.layers.append(GeneratorLayer(prev_size, curr_size, tanh, der_tanh))
            # else:
            #     self.layers.append(GeneratorLayer(prev_size, curr_size, relu, der_relu))
            # self.last_layer += 1
            self.layers.append(GeneratorLayer(prev_size, curr_size, sigmoid, der_sigmoid))

    def generate(self, Z):
        self.layers[0].activated = np.array(Z)
        for i in range(1, len(self.layers)):
            self.layers[i].forwardprop(self.layers[i - 1])
        # return (2*self.layers[-1].activated - 255)/255
        return self.layers[-1].activated

    def backprop(self, Z, D):
        self.layers[-1].activated_der = self.loss_der(Z, D)
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].backprop(self.layers[i - 1], self.learning_rate)

    def loss_der(self, Z, D):
        z = self.generate(Z)
        d = D.predict(z)
        ans = -1 / (1.01 - d)
        D.backprop(z, 0, True)
        ans = ans * np.array(D.layers[0].activated_der)
        return ans

    def save(self, directory):
        filename = directory + '/' + 'gen.weights'
        utils.ensure_directory(directory)
        f = open(filename, 'w')
        for layer in self.layers:
            f.write(' '.join([str(x) for row in layer.weights for x in row]) + '\n')
            f.write(' '.join([str(x) for x in layer.biases]) + '\n')
        f.close()

    def load(self, filename):
        try:
            f = open(filename, 'r')
        except:
            return
        for layer in self.layers:
            line = f.readline().strip('\n').split(' ')
            for row in range(len(layer.weights)):
                for x in range(len(layer.weights[0])):
                    layer.weights[row][x] = float(line[row * len(layer.weights[0]) + x])
            line = f.readline().strip('\n').split(' ')
            for x in range(len(layer.biases)):
                layer.biases[x] = float(line[x])
        f.close()
