import numpy as np
from activantions import *

class GeneratorLayer:
    def __init__(self, prev_size, curr_size, act_f, act_f_der):
        self.W = np.array(np.random.normal(0, 0.1, size=(curr_size, prev_size)))
        self.B = np.array(np.random.normal(0, 0.01, size=(curr_size)))
        self.Z = np.zeros((curr_size))
        self.A = np.zeros((curr_size))

        self.dW = np.zeros((curr_size, prev_size))
        self.dZ = np.zeros((curr_size))
        self.dA = np.zeros((curr_size))

        self.act_f = np.vectorize(act_f)
        self.act_f_der = np.vectorize(act_f_der)

    def __str__(self):
        return str(self.W) + "  " + str(self.B)

    def forwardprop(self, prev_layer):
        self.Z = np.dot(self.W, prev_layer.A) + self.B
        self.A = self.act_f(self.Z)

    def backprop(self, prev_layer,learning_rate):
        self.dZ = self.act_f_der(self.Z)
        self.dB = self.dA*self.dZ
        self.dW = np.outer(self.dB, prev_layer.A)
        self.W = self.W - learning_rate*self.dW
        self.B = self.B - learning_rate*self.dB
        prev_layer.dA = np.dot(np.transpose(self.W), self.dB)


class Generator:
    def __init__(self, learning_rate=0.01, noise_dim=256):
        self.layers = []
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.layers.append(GeneratorLayer(1, noise_dim, sgmd, dsgmd))
        self.layers.append(GeneratorLayer(noise_dim, 1024, sgmd, dsgmd))
        self.layers.append(GeneratorLayer(1024, 1024, sgmd, dsgmd))
        self.layers.append(GeneratorLayer(1024, 784, sgmd, dsgmd))

    def generate(self, Z):
        self.layers[0].A = np.array(Z)
        for i in range(1, len(self.layers)):
            self.layers[i].forwardprop(self.layers[i-1])
        return self.layers[-1].A

    def backprop(self, Z, D):
        self.layers[-1].dA = self.dLoss(Z,D)
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i].backprop(self.layers[i-1],self.learning_rate)
    
    def dLoss(self,Z,D):
        z = self.generate(Z)
        d = D.predict(z)
        ans = -1/(1.01-d)
        D.backprop(z,0,True)
        ans = ans * np.array(D.layers[0].dA)
        return ans

    
    def save(self, filename):
        f = open(filename, 'w')
        for layer in self.layers:
            f.write(' '.join([str(x) for row in layer.W for x in row])+'\n')
            f.write(' '.join([str(x) for x in layer.B])+'\n')
        f.close()

    def load(self, filename):
        try:
            f = open(filename, 'r')
        except:
            return
        for layer in self.layers:
            line = f.readline().strip('\n').split(' ')
            for row in range(len(layer.W)):
                for x in range(len(layer.W[0])):
                    layer.W[row][x] = float(line[row*len(layer.W[0])+x])
            line = f.readline().strip('\n').split(' ')
            for x in range(len(layer.B)):
                layer.B[x] = float(line[x])
        f.close()