#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Layer class: contains a list of neurons

TODO:
    Consider inheriting from this class for HiddenLayer, InputLayer, OutputLayer,
    because:
        - neurons should be a dict for the output layer, but a sequence for the others
        - connect_to() isn't used at all for the input layer
        - update_neurons() isn't used at all for the input layer
    But is it clearer this way ?...
"""

import math

from neuron import Neuron
from synapse import Synapse

class Layer:

    neurons = []
    previousLayer = None
    classLabel = None # Only used if the layer is the output layer

    def __init__(self, nbNeurons, classes = None):
        self.size = nbNeurons
        
        # For the output layer, set  the class labels on the neurons
        if classes is not None:
            self.neurons = {}
            for class_ in classes:
                self.neurons[class_] = Neuron()
        # For every other layer, simply have a sequence of neurons
        else:
            for i in range(0, nbNeurons):
                self.neurons.append(Neuron())


    # Connect this layer to the previous one by connecting synapses to each neurons
    def connect_to(self, previousLayer):
        self.previousLayer = previousLayer
        for neuron in self.neurons:
            for previousNeuron in self.previousLayer.neurons:
                neuron.connect_to(previousNeuron)


    # Squish the value into the interval [0,1]
    def squish(self, value):
        # use the sigmoid function
        return 1 / (1 + math.exp(-value))


    # Updates the value of each neuron
    def update_neurons(self):
        for neuron in self.neurons:
            value = 0
            for synapse in neuron.synapses:
                value += synapse.neuronFrom.value * synapse.weight

            value -= neuron.bias
            neuron.set_value(squish(value))


