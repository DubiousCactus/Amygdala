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
        self.bias = 0
        self.neurons = []
        self.size = nbNeurons
        self.classLabel = None # Only used if the layer is the output layer

        for i in range(nbNeurons):
            self.neurons.append(Neuron())


    # For the output layer, set the class labels on the neurons
    def set_class_labels(self, classLabels):
        self.neurons = {}
        for class_ in classLabels:
            self.neurons[class_] = Neuron()


    # Connect this layer to the next one by connecting synapses to each neurons
    def connect_to(self, nextLayer):
        if type(nextLayer.neurons) is dict: # For the output layer
            nextNeurons = list(nextLayer.neurons.values())
        else: # For the other layers
            nextNeurons = nextLayer.neurons

        for neuron in self.neurons:
            for nextNeuron in nextNeurons:
                neuron.connect_to(nextNeuron)
                nextNeuron.connect_from(neuron)


    # Activation function: squish the value into the interval [0,1] using the sigmoid function
    def squish(self, value):
        return 1.0 / (1.0 + np.exp(-value))


    # Updates the value of each neuron
    def feed_forward(self):
        if type(self.neurons) is dict: # For the output layer
            neurons = list(self.neurons.values())
        else: # For the other layers
            neurons = self.neurons

        for neuron in neurons:
            value = 0
            for synapse in neuron.synapses_from:
                value += synapse.neuron.value * synapse.weight

            neuron.set_value(self.squish(value + self.bias))


