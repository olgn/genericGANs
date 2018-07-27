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
this currently runs the mnist gan example as laid out in https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

the program is run via
```
python gan.py
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
