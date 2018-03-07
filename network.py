#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 transpalette <transpalette@arch-cactus>
#
# Distributed under terms of the MIT license.

"""
Network class: assembles layers and implements the error correction / weight adjustment algorithms
"""

import sys
import random
import numpy as np

from PIL import Image
from layer import Layer

class Network:

    learningRate = 0
    inputLayer = None
    outputLayer = None
    hiddenLayers = []
    inputs = {}
    trainingData = {}
    testingData = []

    # Creates and inits layers
    def __init__(self, nbPixels, nbClasses, learningRate):
        self.inputLayer = Layer(nbPixels)
        self.outputLayer = Layer(nbClasses)
        self.learningRate = learningRate


    def add_hidden_layer(self, size):
        hiddenLayers.append(Layer(size))


    # Initialize the input layer's neurons
    # Input format:{
    #    'class': [[ pixelVal, pixelVal, ... ], [ ... ], ... ],
    #    ...
    # }
    def set_inputs(self, inputs):
        try:
            if len(next(iter(inputs.values()))[0]) != self.inputLayer.size:
                raise AssertionError("Input size doesn't match")
        except AssertionError as error:
            print("Error caught: " + repr(error))
            sys.exit(1)

        self.inputs = inputs
        self.outputLayer.set_class_labels(self.inputs.keys())
        self.split_data()


    # Split the input data into 80% training and 20% testing
    def split_data(self):
        print("[*] Splitting input elements")
        # Loop through each class and shuffle the inputs
        for class_, inputs in self.inputs.items():
            # Discard elemets above 75K index
            # Trying with this first:
            inputs = np.delete(inputs, np.s_[::2], 0)

            print("\t-> Selecting training/testing data for class: {}".format(class_))
            random.shuffle(inputs)

            self.trainingData[class_] = []
            # Take the first 80% elements to use them as training data
            for i in range(0, int(round(0.8 * len(inputs)))):
                self.trainingData[class_].append(
                    inputs[i] / 255 # Normalize values to [0,1]
                )

            # The rest is of course the test data
            for i in range(int(round(0.8 * len(inputs)) + 1), len(inputs)):
                self.testingData.extend(
                    inputs[i] / 255 # Normalize values to [0,1]
                )


        print("[*] Shuffling testing data")
        # Shuffle the test data
        random.shuffle(self.testingData)
        # Clear the inputs, they aren't need anymore
        del self.inputs


    def get_output(self):
        return


    def back_propagate(self):
        return


    def mean_square_error(self):
        return


    def train(self):
        expectedOutputs = []
        for class_, elements in self.trainingData:
            for element in elements:
                # Setting the input neuronns' value to the pixels' value of the current
                # element of the current class
                for i in range(0, self.inputLayer.size):
                    self.inputLayer.neurons[i].set_value(element[i])
                
                # Set the expected output layer's outputs' values accordingly
                for outputNeuron in self.outputLayer.neurons:
                    if outputNeuron.classLabel == class_:
                        expectedOutputs[i] = 1
                    else:
                        expectedOutputs[i] = -1

                # Run the neural network for the current input
                for layer in self.hiddenLayers:
                    layer.update_neurons()

                self.outputLayer.update_neurons()

                # Adjust weights
                self.back_propagate() # or self.mean_square_error()



    def classify(self):
        return
    


if __name__ == "__main__":
    random.seed()
    # Using npz files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/
    print("[*] Loading data sets")
    dataSets = {
        'swords': np.load('datasets/full_numpy_bitmap_sword.npy'),
        'skulls': np.load('datasets/full_numpy_bitmap_skull.npy'),
        'skateboards': np.load('datasets/full_numpy_bitmap_skateboard.npy'),
        'pizzas': np.load('datasets/full_numpy_bitmap_pizza.npy')
    }

    # testImage = Image.fromarray(dataSets['swords'][0].reshape(28, 28))
    # testImage.resize((600, 600)).show()
    
    neuralNetwork = Network(28*28, len(dataSets), 5)
    neuralNetwork.set_inputs(dataSets)
    del dataSets
    # neuralNetwork.train()
    
    # ...

