from gan import GAN
from utils import load_data, show_img, save_img
import numpy as np
import random
from pathlib import Path

Xline, Yline = load_data('line.txt')
Xtest, Ytest = load_data('fashion_mnist_test.csv')


def test_train(relevant):
    X = []
    for i in range(len(Xtest)):
        if Ytest[i] in relevant:
            X.append(Xtest[i])
    G = GAN(X)

    for x in range(30):
        G.train(500)
        ans_gen = 0
        ans_data = 0
        for i in range(20):
            ans_gen += G.D.predict(G.generate())
            ans_data += G.D.predict(random.choice(X))
        print(ans_gen/20, ans_data/20, G.D.predict(Xline[0]), G.D.learning_rate, G.G.learning_rate)
        rel_rep = '_'.join(map(str, relevant))
        dir_path = 'renders/' + rel_rep
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        save_img(G.generate(), dir_path + '/Iteration'+str(x).zfill(2), x)
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


for i in range(10):
    while True:
        try:
            test_train([i])
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            pass
# test_gen()
# test_disc()
