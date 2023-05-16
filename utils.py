import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    X = []
    Y = []

    with open(filename, 'r') as f:
        for line in f:
            l = line.split(',')
            temp = int(l[0])
            Y.append(temp)
            X.append([int(x)/255 for x in l[1:]])
    return X, Y

def show_img(V):
    V = np.array(V)
    for i in range(len(V)):
        V[i] *= 255
    plt.imshow(V.reshape((28,28)))
    plt.gray()
    plt.show()
