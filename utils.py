import pathlib

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
    plt.imshow(V.reshape((28, 28)))
    plt.gray()
    plt.show()


def save_img(V, directory, filename):
    V = np.array(V)
    for i in range(len(V)):
        V[i] *= 255
    plt.imshow(V.reshape((28, 28)))
    plt.gray()
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{directory}/{filename}')


def generate_line():
    for r in range(2):
        s = '0'
        for x in range(28):
            for y in range(28):
                if y > 10 and y < 17:
                    s += ',0'
                else:
                    s += ',255'
        print(s)
# generate_line()
