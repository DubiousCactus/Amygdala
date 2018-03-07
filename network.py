#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Network class: assembles layers and implements the error correction / weight adjustment algorithms
"""

import random
import numpy as np

from PIL import Image
from layer import Layer

class Network:

    learningRate = 0
    inputLayer = None
    outputLayer = None
    hiddenLayers = []
    inputs = []

    # Creates and inits layers
    def __init__(self, nbPixels, nbClasses, learningRate):
        self.inputLayer = Layer(nbPixels)
        self.outputLayer = Layer(nbClasses)
        self.learningRate = learningRate


    def add_hidden_layer(self, size):
        hiddenLayers.append(Layer(size))


    # Initialize the input layer's neurons
    # Input format: [[ pixelVal, pixelVal, ... ], [ ... ], ... ]
    def set_inputs(self, inputs):
        try:
            if length(inputs[i]) != self.inputLayer.size:
                raise ValueException("Input size doesn't match")
        except ValueError as error:
            print("Error caught: " + repr(error))

        self.inputs = inputs


    def get_output(self):
        return


    def back_propagate(self):
        return


    def mean_square_error(self):
        return


    def train(self):
        for input in self.inputs:
            for i in range(0, self.inputLayer.size):
                # Remember to normalize the inputs !
                self.inputLayer.neurons[i].set_value(inputs[i])
            
            # init class vectors's values to -1 or 1 depending on the test class


            # Run the neural network for the current input
            for layer in self.hiddenLayers:
                layer.update_neurons()

            self.outputLayer.update_neurons()

            # Adjust weights
            self.back_propagate() # or self.mean_square_error()



    def classify(self):
        return
    
    def augment_data():
        return



if __name__ == "__main__":
    random.seed()
    # Using npz files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn
    dataSets = {
        'swords': np.load('datasets/full_numpy_bitmap_sword.npy'),
        'skulls': np.load('datasets/full_numpy_bitmap_skull.npy'),
        'skateboards': np.load('datasets/full_numpy_bitmap_skateboard.npy'),
        'pizzas': np.load('datasets/full_numpy_bitmap_pizza.npy')
    }

    testImage = Image.fromarray(dataSets['swords'][0].reshape(28, 28))
    testImage.resize((600, 600)).show()
    
    # neuralNetwork = Network(dataset.nbPixels, dataset.nbClasses, 5)
    # neuralNetwork.set_input(dataSets)
    # neuralNetwork.train()
    
    # ...

