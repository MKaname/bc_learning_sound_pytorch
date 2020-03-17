import torch

class Transform(object):
    def __init__(self, opt):
        self.opt = opt
    def __call__(self, sample):
        print(sample.shape)
        return sample.reshape(1, 1, sample.shape[-1])
