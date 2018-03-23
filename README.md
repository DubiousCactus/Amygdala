# ! WIP !
**This is still a work in progress, and does not offer a graphical user interface for now.**

# Introduction
The *Amygdala* corresponds to the part of the brain that is responsible for the processing of memory and the making of decisions, kind of like a neural network...

This project aims to demonstrate how deep neural networks (a perceptron in this example) work through a simple, clean, and easy to understand code structure, in object oriented to avoid the matrix headaches. A (soon to come) visualisation GUI also helps understanding how the data is processed within such structure, and parameters can be adjusted in order to understand from the basics, to more advanced calculations. Of course, everything can be greatly optimized with the use of matrices, but the goal is not to have a fast and very precise Deep Neural Network, but rather to offer a clear overview of the implementation of one.

It is probably much slower than your quickly written TensorFlow NN, but guarenties satisfying results :) (and even your kids can read the code !)

# Deep learning
This project implements a Deep Neural Network, or a multi-layer perceptron, with a few hidden layers.
## Optimization algorithms
 * Back-propagation

## Data set
This Neural Network is trained and tested on Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) data set. The `.pny` **Numpy** files are not included in this repo, therefore you have to download the ones you want to use from [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) (I recommend using about 4-5 classes/files).
![preview](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/preview.jpg?raw=true)


# Todo

 - [x] Write code base
 - [x] Write the feed forward algorithm
 - [x] Write the backpropagation algorithm
 - [ ] Code the GUI
