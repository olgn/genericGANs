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

# CIFAR 10 DATA
# resizes data to 64 x 64, then transforms to tensor, then normalizes to mean .5 + std dev .5
def cifar_data():
    # resize, transform, and normalize data
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
    )
    # download or reference the pytorch cifar10 dataset
    out_dir = './dataset'
    data = datasets.CIFAR10(root=out_dir, train=True, transform=compose, download=True)
    return data


# Creates the DataLoader that will generate data from the ./dataset folder
def get_data_loader(data):
    # Create loader with data
    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

    return data_loader

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

# Converts images to vectors
def images_to_vectors(images, size):
    return images.view(images.size(0), size)

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