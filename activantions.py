import numpy as np


def ReLU(x):
    return max(0, x)


def dReLU(x):
    if x <= 0:
        return 0
    else:
        return 1


def sgmd(x):
    if x > 10:
        return 1
    if x < -10:
        return 0
    return (1/(1+(np.e**(-x))))


def dsgmd(x):
    return (sgmd(x)*(1-sgmd(x)))


def id(x):
    return x


def sftmx(A):
    exps = np.exp(A - A.max())
    return exps / np.sum(exps, axis=0)


def dsftmx(A):
    exps = np.exp(A - A.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
