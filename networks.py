# ml tools
import torch
from torch import nn
from torchvision import transforms, datasets

class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 784 # mnist images are 28x28=784
        n_out=1

        # first layer is a linear nn that expands 784 input nodes to 1024 output nodes,
        # then uses a leakyRelu activation function with a negative slope of 0.2,
        # and a dropout layer with a 30% chance of dropout
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # second layer is a linear nn that reduces 1024 input nodes to 512 output nodes,
        # then uses a leakyRelu activation function with a negative slope of 0.2,
        # and a dropout layer with a 30% chance of dropout
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        # third layer is a linear nn that reduces 512 input nodes to 256 output nodes,
        # then uses a leakyRelu activation function with a negative slope of 0.2,
        # and a dropout layer with a 30% chance of dropout
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        # output layer is a linear nn that maps 256 input nodes to 1 output nodes,
        # and runs that through a sigmoid activation function in order to provide a [0,1] continous
        # output value
        self.out = nn.Sequential(
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
class Generator(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(Generator, self).__init__()
        n_features = 100
        n_out = 784

        # first layer is a linear nn that expands 100 noise input nodes to 256 output nodes,
        # then uses a leakyRelu activation function with a negative slope of 0.2
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        # second layer is a linear nn that expands 256 noise input nodes to 512 output nodes,
        # then uses a leakyRelu activation function with a negative slope of 0.2
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        # third layer is a linear nn that expands 512 noise input nodes to 1024 output nodes,
        # then uses a leakyRelu activation function with a negative slope of 0.2
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        # output layer is a linear nn that reduces 1024 input nodes to 784 output nodes,
        # and runs that through a tanh activation function in order to provide a [-1,1] continous
        # output value
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    