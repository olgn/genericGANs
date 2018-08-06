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
from networks import AudioDiscriminator, AudioGenerator

# optimizers
from optimizers import linear_adam_optimizer

# training steps
from train import train_discriminator, train_generator

sample_size = 160000 # length of audio samples used to train

# load the vctk utterance data
vctk = data.vctk_data(max_len=sample_size)
data_loader = data.get_data_loader(vctk, 100)

# get the number of batches from the data loader
num_batches = len(data_loader)

# create the discriminator model + optimizer
discriminator = AudioDiscriminator(sample_size)
d_optimizer = linear_adam_optimizer(discriminator.parameters(), 0.0002)

# create the generator model + optimizer
generator = AudioGenerator(sample_size)
g_optimizer = linear_adam_optimizer(generator.parameters(), 0.0002)

# create loss function
loss = nn.BCELoss()

# establish some test noise to feed to the generator throughout training
num_test_samples = 16
test_noise = data.noise(num_test_samples)


# train
def train():
    logger = Logger(model_name="AGAN", data_name="VCTK")
    num_epochs = 200
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)

            # TRAIN DISCRIMINATOR
            # generate real data from data loader
            real_data = Variable(real_batch.view(real_batch.size(0), sample_size))

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
            
            print('epoch:', epoch, 'n_batch:', n_batch)
            # Display progress every few batches
            if (n_batch) % 10 == 0:
                test_audio = generator(test_noise).data
                print(test_audio.size())

if __name__ == "__main__":
    train()