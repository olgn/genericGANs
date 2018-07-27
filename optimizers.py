import torch
from torch import optim

def adam_optimizer(params, learning_rate):
    return optim.Adam(params, lr=learning_rate)
