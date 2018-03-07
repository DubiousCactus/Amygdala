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

    # Creates and inits layers
    def __init__(self, nbNeurons, nbHiddenLayers, learningRate):
        self.nbNeurons = nbNeurons
        self.inputLayer = Layer(nbNeurons)
        self.outputLayer = Layer(nbNeurons)

        for i in range(0, nbHiddenLayers):
            self.hiddenLayers.append(Layer(nbNeurons))

        self.learningRate = learningRate


    # Initialize the input layer's neurons
    # Input format: [ pixelVal, pixelVal, ... ]
    def set_input(self, inputs):
        try:
            if length(inputs) != self.nbNeurons:
                raise ValueException("Input size doesn't match")
        except ValueError as error:
            print("Error caught: " + repr(error))

        for i in range(0, nbNeurons):
            # Remember to normalize the inputs !
            inputLayer.neurons[i].set_value(inputs[i])


    def get_output(self):


    def back_propagate(self):


    def mean_square_error(self):


    def train(self):
        # init output layer's neurons to -1 or 1 depending on test data


    def classify(self):

    
    def augment_data():



if __name__ == "__main__":
    random.seed()
    neuralNetwork = Network()
    # ...

