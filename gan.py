from gen_mod import Generator
from dis_mod import Discriminator
import numpy as np
import random
from utils import show_img
from config import *

class GAN:
    def __init__(self,data,noise_dim=512):
        self.noise_dim = noise_dim
        self.D = Discriminator(0.05)
        self.G = Generator(0.1,noise_dim)
        self.data = data
    
    def train(self,iters = 100, K = 1, lr_mod=0):
        #self.D.learning_rate *= lr_mod
        #self.G.learning_rate -= lr_mod
        #self.G.learning_rate = max(self.G.learning_rate,0.02)
        for it in range(iters):
            if it % 100 == 0:
                ans_gen = 0
                ans_data = 0
                #loss = 0
                for i in range(20):
                    pg = self.D.predict(self.generate())
                    pd = self.D.predict(random.choice(self.data))
                    #loss += np.log(pd) + np.log(1-pg)
                    ans_gen += pg#+0.05
                    ans_data += pd
                ans_gen /= 20
                ans_data /= 20

                #K = 1
                #if(ans_data < 0.4 or ans_gen > 0.6):
                #    K += 3
                #if(ans_gen < 0.2):
                if(ans_data > 0.9 or ans_data-ans_gen > 0.3):
                    K = 0
                if(ans_data-ans_gen < 0.4):
                    K = 1
                if(ans_gen > ans_data):
                    K = 3
                if(ans_data < 0.3):
                    K = 5
                K = 1
                #print(loss/20)
                #ans = ans_gen-ans_data
                #ans /= 20
                #ans *= 0.01
                #self.D.learning_rate += ans/10
                #self.D.learning_rate = max(self.D.learning_rate,0.001)
                #self.G.learning_rate -= ans * 0.5 * self.G.learning_rate + 0.001
                #self.G.learning_rate = max(0,self.G.learning_rate)
            for k in range(K):
                z = np.random.normal(size=self.noise_dim,scale=NOISE_SCALE)
                x = random.choice(self.data)
                self.D.backprop(x,1)
                self.D.backprop(self.G.generate(z),0)
            for i in range(3):
                z = np.random.normal(size=self.noise_dim,scale=NOISE_SCALE)
                self.G.backprop(z,self.D)
    def generate(self):
        return self.G.generate(np.random.normal(size=self.noise_dim,scale=NOISE_SCALE))
    
    def save(self,gen_filename='gen.weights',discrim_filename='discrim.weights'):
        self.G.save(gen_filename)
        self.D.save(discrim_filename)
    
    def load(self,gen_filename='gen.weights',discrim_filename='discrim.weights'):
        self.G.load(gen_filename)
        self.D.load(discrim_filename)
