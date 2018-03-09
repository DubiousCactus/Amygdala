#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Neuron class
"""

from synapse import Synapse

class Neuron:


    def __init__(self):
        self.value = 0
        self.bias = 0
        self.synapses = [] # A synapse is only contained in the neuron if its layer has a previous layer


    def set_value(self, value):
        self.value = value


    # Add a (positive or negative) value to the bias
    def update_bias(self, value):
        self.bias += value


    def connect_to(self, previousNeuron):
        self.synapses.append(Synapse(previousNeuron, self))
