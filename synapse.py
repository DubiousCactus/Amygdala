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


class Synapse:
    
    def __init__(self, from_, to):
        self.neuronFrom = from_
        self.neuronTo = to
        self.weight = random.random() # [0.0, 1.0)

    
    # Adds a (positive or negative) value to the weight
    def update_weight(self, value):
        self.weight += value
