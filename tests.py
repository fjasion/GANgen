from gan import GAN
from utils import load_data,show_img
import numpy as np

#Xtr, Ytr = load_data('fashion-mnist_train.csv')
Xtest, Ytest = load_data('line.txt')
X = []
def test():
    for i in range(len(Xtest)):
        if Ytest[i] == 0:
            X.append(Xtest[i])
    G = GAN(X)
    show_img(X[0])
    G.train(30000)
    show_img(G.generate())

test()

