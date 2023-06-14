import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np

import config


def data_path(filename):
    return config.DATA_DIRECTORY + '/' + filename


def load_data(filename):
    X = []
    Y = []

    with open(data_path(filename), 'r') as f:
        for line in f:
            l = line.split(',')
            temp = int(l[0])
            Y.append(temp)
            # X.append([(2*int(x) - 255)/255 for x in l[1:]])
            X.append([(int(x)) / 255 for x in l[1:]])
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
    ensure_directory(directory)
    plt.savefig(f'{directory}/{filename}')


def ensure_directory(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def filter_by_labels(X, Y, labels):
    Y = np.array(Y)
    present = np.full(len(X), False)
    for label in labels:
        present |= Y == label
    return np.array(X)[present]


def log_time(f, description, level):
    print(level * '  ' + 'Starting:', description)
    start = time.time()
    result = f()
    end = time.time()
    print(level * '  ' + 'Done:', description + ', took', end - start, 'seconds')
    return result


def generate_line():
    for r in range(2):
        s = '0'
        for x in range(28):
            for y in range(28):
                if 10 < y < 17:
                    s += ',0'
                else:
                    s += ',255'
        print(s)
# generate_line()
