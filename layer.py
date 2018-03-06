#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Layer class: contains a list of neurons
"""

import math

from neuron import Neuron
from synapse import Synapse

class Layer:

    neurons = []
    previousLayer = None

    def __init__(self, nbNeurons):
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
        # use sigmoid
        return 1 / (1 + math.exp(-value))


    # Updates the value of each neuron
    def update_neurons(self):
        for neuron in self.neurons:
            value = 0
            for synapse in neuron.synapses
                value += synapse.neuronFrom.value * synapse.weight

            value -= neuron.bias
            neuron.set_value(squish(value))


