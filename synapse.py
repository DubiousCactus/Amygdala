#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Synapse class: transfers data between neurons.
"""

import random

from neuron import Neuron


class Synapse:
    
    def __init__(self, from, to):
        self.neuronFrom = from
        self.neuronTo = to
        self.weight = random.random() # [0.0, 1.0)


    def set_weight(self, weight):
        self.weight = weight


