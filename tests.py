from gan import GAN
from utils import load_data,show_img
import numpy as np

#Xtr, Ytr = load_data('fashion-mnist_train.csv')
Xtest, Ytest = load_data('fashion-mnist_test.csv')

def test():
    X = []
    for i in range(len(Xtest)):
        if Ytest[i] == 0:
            X.append(Xtest[i])
    G = GAN(X)
    G.train(100000)
    show_img(G.generate())

test()