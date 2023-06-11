from gan import GAN
from utils import load_data, show_img, save_img
import numpy as np
import random

import config

Xline, Yline = load_data('line.csv')
Xtest, Ytest = load_data('fashion_mnist_test.csv')
X = []


def test_train():
    for i in range(len(Xtest)):
        if Ytest[i] == 9:
            X.append(Xtest[i])
    G = GAN(X, gen_layer_sizes=[512, 1024, 1024, 784])

    for x in range(1):
        G.train(1000, lr_mod=0.01)
        ans_gen = 0
        ans_data = 0
        for i in range(20):
            ans_gen += G.D.predict(G.generate())
            ans_data += G.D.predict(random.choice(X))
        print(ans_gen/20, ans_data/20, G.D.predict(Xline[0]), G.D.learning_rate, G.G.learning_rate)
        save_img(G.generate(), 'renders/render15', 'Iteration'+str(x))
        G.save()


def test_gen():
    G = GAN(X)
    G.load()
    for i in range(10):
        show_img(G.generate())


def test_disc():
    G = GAN(X)
    G.load()
    z = G.generate()
    print(G.D.predict(z))
    print(G.D.predict(Xtest[3]))
    print(G.D.predict(Xline[0]))

def setup():
    np.random.seed(config.SEED)
    random.seed(config.SEED)


def main():
    test_train()
    # test_gen()
    # test_disc()

if __name__ == '__main__':
    setup()
    main()
