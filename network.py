#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Network class: assembles layers and implements  the backpropagation algorithm
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
        self.inputLayer = Layer(nbNeurons)
        self.outputLayer = Layer(nbNeurons)

        for i in range(0, nbHiddenLayers):
            self.hiddenLayers.append(Layer(nbNeurons))

        self.learningRate = learningRate


    # Initialize the input layer's neurons
    def set_input(self, inputs):


    def get_output(self):


    def back_propagate(self):


    def train(self):
        # init output layer's neurons to -1 or 1 depending on test data


    def classify(self):

    
    def augment_data():



if __name__ == "__main__":
    random.seed()
    neuralNetwork = Network()
    # ...

