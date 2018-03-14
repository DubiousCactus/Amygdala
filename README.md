# ! WIP !
**This is still a work in progress, and does not offer anything functionnal for now.**

# Introduction
The *Amygdala* corresponds to the part of the brain that is responsible for the processing of memory and the making of decisions.

This project aims to demonstrate how neural networks work through a simple, clean, and easy to understand code structure. A (soon to come) visualisation GUI also helps understanding how the data is processed within such structure, and parameters can be adjusted in order to understand from the basics, to more advanced calculations. Of course, everything can be greatly optimized with the use of matrices, but the goal is not to have a fast and very precise Deep Neural Network, but rather to offer a clear overview of the implementation of one.


# Deep learning
This project implements a Deep Neural Network, or a multi-layer perceptron, with a few hidden layers.
## Optimization algorithms
 * Back-propagation
 * Minimum square error

## Data set
This Neural Network is trained and tested on Google's [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) data set. The `.pny` **Numpy** files are not included in this repo, therefore you have to download the ones you want to use from [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) (I recommend using about 4-5 classes/files).
![preview](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/preview.jpg?raw=true)


# Todo

 - [x] Write code base
 - [x] Write the feed forward algorithm
 - [ ] Write the backpropagation algorithm
 - [ ] Code the GUI
