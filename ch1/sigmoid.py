import numpy as np

class Sigmid:
    def __init__(self):
        self.params = []

    def forward(self,x):
        return 1/(1+np.exp(-x)) 



class Affine:
    def __init__(self,w,b):
        self.params=[w,b]

    def forward(self,x):
        w,b = self.params
        out = np.dot(x,w) + b
        return out
