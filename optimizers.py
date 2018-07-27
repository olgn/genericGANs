import torch
from torch import optim

def linear_adam_optimizer(params, learning_rate):
    return optim.Adam(params, lr=learning_rate)


def dc_adam_optimizer(params, learning_rate):
    return optim.Adam(params, lr=learning_rate, betas=(0.5, 0.999))
