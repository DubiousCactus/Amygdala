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


    def back_propagate(self):


    def mean_square_error(self):


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

    
    def augment_data():



if __name__ == "__main__":
    random.seed()
    dataset = DataSet(args[1])
    neuralNetwork = Network(dataset.nbPixels, dataset.nbClasses, 5)
    neuralNetwork.set_input(dataset.parse_inputs())
    neuralNetwork.train()
    
    # ...

