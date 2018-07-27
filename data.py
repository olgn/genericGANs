import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

# MNIST data import function.
# transforms data to tensor, then normalize to
# mean .5 + std dev .5
def mnist_data():
    # transform + normalize data
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
    )

    # download or reference the pytorch mnist dataset
    out_dir = './dataset'
    data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

    return data

# Creates the DataLoader that will generate data from the ./dataset folder
def get_data_loader():
    # Load data
    data = mnist_data()
    
    # Create loader with data
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

    return data_loader

# Converts images to vectors
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# Converts vectors to images
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
    """
    Generates a 1-d vector of gaussian noise
    """
    n = Variable(torch.randn(size, 100))
    return n

def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    return data