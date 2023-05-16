from gen_mod import Generator
from dis_mod import Discriminator
import numpy as np
import random
from utils import show_img

class GAN:
    def __init__(self,data):
        self.D = Discriminator()
        self.G = Generator()
        self.data = data
    
    def train(self,iters = 100, K = 1):
        for it in range(iters):
            if it % 10000 == 0:
                show_img(self.G.generate(np.random.normal(size=32)))
            for k in range(K):
                z = np.random.normal(size=32)
                x = random.choice(self.data)
                self.D.backprop(x,1)
                self.D.backprop(self.G.generate(z),0)
            z = np.random.normal(size=32)
            self.G.backprop(z,self.D)
    def generate(self):
        return self.G.generate(np.random.normal(size=32))
