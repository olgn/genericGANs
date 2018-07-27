# ml tools
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

# logging tools
from utils import Logger

# data tools
import data

# network models
from networks import Discriminator, Generator

# optimizers
from optimizers import adam_optimizer

# training steps
from train import train_discriminator, train_generator

# get the data loader
data_loader = data.get_data_loader()

# get the number of batches from the data loader
num_batches = len(data_loader)

# create the discriminator model + optimizerr
discriminator = Discriminator()
d_optimizer = adam_optimizer(discriminator.parameters(), 0.0002)

# create the generator model + optimizer
generator = Generator()
g_optimizer = adam_optimizer(generator.parameters(), 0.0002)

# create loss function
loss = nn.BCELoss()

# establish some test noise to feed to the generator throughout training
num_test_samples = 16
test_noise = data.noise(num_test_samples)

# train
def train():
    logger = Logger(model_name="VGAN", data_name="MNIST")
    num_epochs = 200
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)

            # TRAIN DISCRIMINATOR
            # generate real data from data loader
            real_data = Variable(data.images_to_vectors(real_batch))

            # generate fake data and detach gradient
            fake_data = generator(data.noise(N)).detach()

            # train discriminator
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator, loss, d_optimizer, real_data, fake_data)

            # TRAIN GENERATOR

            # generate fake data
            fake_data = generator(data.noise(N))

            # train generator
            g_error = train_generator(discriminator, loss, g_optimizer, fake_data)

            # LOG BATCH ERROR  
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display progress every few batches
            if (n_batch) % 100 == 0:
                test_images = data.vectors_to_images(generator(test_noise))
                test_images = test_images.data

                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)
                logger.display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake)

