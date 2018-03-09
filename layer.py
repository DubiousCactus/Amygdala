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

import numpy as np

from neuron import Neuron
from synapse import Synapse

class Layer:


    def __init__(self, nbNeurons):
        self.neurons = []
        self.size = nbNeurons
        self.previousLayer = None
        self.classLabel = None # Only used if the layer is the output layer

        for i in range(nbNeurons):
            self.neurons.append(Neuron())


    # For the output layer, set the class labels on the neurons
    def set_class_labels(self, classLabels):
        self.neurons = {}
        for class_ in classLabels:
            self.neurons[class_] = Neuron()


    # Connect this layer to the previous one by connecting synapses to each neurons
    def connect_to(self, previousLayer):
        self.previousLayer = previousLayer
        for neuron in self.neurons:
            for previousNeuron in self.previousLayer.neurons:
                neuron.connect_to(previousNeuron)


    # Squish the value into the interval [0,1]
    def squish(self, value):
        # use the sigmoid function
        return 1 / (1 + np.exp(-value))


    # Updates the value of each neuron
    def feed_forward(self):
        if type(self.neurons) is dict: # For the output layer
            neurons = list(self.neurons.values())
        else: # For the other layers
            neurons = self.neurons

        for neuron in neurons:
            value = 0
            for synapse in neuron.synapses:
                value += synapse.neuronFrom.value * synapse.weight

            neuron.set_value(self.squish(value + neuron.bias))


