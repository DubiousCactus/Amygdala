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

import numpy as np


class Synapse:
    
    def __init__(self, from_, to):
        self.neuronFrom = from_
        self.neuronTo = to
        self.weight = (-1.0 - 1.0) * np.random.random_sample() + 1.0 # [-1.0, 1.0]
        self.updatedWeight = 0

    
    # Adds a (positive or negative) value to the weight
    def update_weight(self, value):
        self.weight += value
