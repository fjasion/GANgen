import numpy as np


def relu(x):
    return np.max(0, x)


def der_relu(x):
    if x <= 0:
        return 0
    else:
        return 1


def leaky_relu(x):
    if x >= 0:
        return x
    else:
        return 0.01 * x


def der_leaky_relu(x):
    if x >= 0:
        return 1
    else:
        return 0.01


def tanh(x):
    return np.tanh(x)


def der_tanh(x):
    return 1 - tanh(x) * tanh(x)


def sgmd(x):
    if x > 10:
        return 1
    if x < -10:
        return 0
    return 1 / (1 + (np.e ** (-x)))


def dsgmd(x):
    return sgmd(x) * (1 - sgmd(x))


def id(x):
    return x


def sftmx(A):
    exps = np.exp(A - A.max())
    return exps / np.sum(exps, axis=0)


def dsftmx(A):
    exps = np.exp(A - A.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))