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
        self.synapses_from = [] # Those synapses lead to the neurons of the next layer
        self.synapses_to = [] # Those synapses lead to the neurons of the previous layer


    def set_value(self, value):
        self.value = value


    # Add a (positive or negative) value to the bias
    def update_bias(self, value):
        self.bias += value


    def connect_to(self, nextNeuron):
        self.synapses_to.append(Synapse(nextNeuron))

    def connect_from(self, previousNeuron):
        self.synapses_from.append(Synapse(previousNeuron))
