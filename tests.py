import traceback

from gan import GAN
from utils import load_data, show_img, save_img, filter_by_labels, log_time
import numpy as np
import random

import config

Xline, Yline = load_data('line.csv')
Xtest, Ytest = load_data('fashion_mnist_test.csv')
X = []


def test_with_discriminator(dataset, labels, num_inner):
    dis_layer_config = [256] * num_inner + [128]
    G = GAN(
        dataset,
        gen_layer_sizes=[512, 1024, 1024],
        gen_learning_rate=0.12,
        dis_layer_sizes=dis_layer_config,
        dis_learning_rate=0.02
    )
    for x in range(15):
        G.train(1000, lr_mod=0.01)
        ans_gen = 0
        ans_data = 0
        for _ in range(20):
            ans_gen += G.D.predict(G.generate())
            ans_data += G.D.predict(random.choice(dataset))
        print('     ', ans_gen/20, ans_data/20, G.D.learning_rate, G.G.learning_rate)
        label_description = 'labels_' + '_'.join(map(str, labels))
        save_img(G.generate(), f'renders/render_{label_description}_inner_{num_inner}', 'iteration'+str(x))


def test_with_dataset(x, y, labels):
    for i in range(1, 4):
        dataset = filter_by_labels(x, y, labels)
        try:
            print(f'    testing {labels} with discriminator with {i} hidden layers')
            log_time(lambda: test_with_discriminator(dataset, labels, i), f'discriminator {i}', 2)
        except:
            print('      This test has failed')
            print('      Probably numpy decided to give up')
            traceback.print_exc()
            print('      This is incredibly sad')


def test_all():
    for labels in [[1], [5], [8], [2, 3], [4, 6], [7, 9]]:
        log_time(lambda: test_with_dataset(Xtest, Ytest, labels), f'testing with labels {labels}', 1)


def test_train():
    for i in range(len(Xtest)):
        if Ytest[i] == 9:
            X.append(Xtest[i])
    G = GAN(
        X,
        gen_layer_sizes=[512, 1024, 1024],
        gen_learning_rate=0.12,
        dis_layer_sizes=[256, 128],
        dis_learning_rate=0.02
    )

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
    log_time(test_all, 'full test', 0)
    # test_train()
    # test_gen()
    # test_disc()


if __name__ == '__main__':
    setup()
    main()
