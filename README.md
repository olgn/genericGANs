# genericGANs
generic implementation of general adversarial networks in python


# Getting Started
create a virtual environment and install package dependencies using the following code:
```
virtualenv .env --python python3.6
source .env/bin/activate
pip install -r requirements.txt
```


# Running the MNIST example
if you would like to explore the MNIST linear GAN example as laid out in https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

the program is run via
```
python lgan.py
```

# Running the CIFAR10 example
included is a working example of deep convolutional GANs on the CIFAR10 dataset as laid out in https://github.com/diegoalejogm/gans

the program is run via
```
python dcgan.py
```

# Visualizing the progress of the algorithm
this software uses the tensorboard implementation for pytorch, tensorboardX, to visualize the different parameters involved in training.
while the program is executing, you may visualize the progress / development via tensorboard by
running the following code:
```
tensorboard --logdir runs
```
and navigating to `localhost:6006` in your web browser.

# Happy hacking!
this project is under development, and will become extensible to data types of various shapes + allow for model flexibility
