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
import math
import random
import progressbar
import numpy as np

from layer import Layer

class Network:

    totalError = 0
    learningRate = 0
    inputLayer = None
    outputLayer = None
    hiddenLayers = []
    inputs = {}
    trainingData = []
    testingData = []
    connected = False

    # Creates and inits layers
    def __init__(self, nbPixels, samplesPerClass, nbClasses, learningRate):
        self.inputLayer = Layer(nbPixels)
        self.outputLayer = Layer(nbClasses)
        self.learningRate = learningRate
        self.samplesPerClass = samplesPerClass
        self.expectedOutputs = []


    def add_hidden_layer(self, size):
        try:
            if self.connected:
                raise Exception("The network is already connected")
        except Exception as error:
            print("Error caught: " + repr(error))

        self.hiddenLayers.append(Layer(size))



    # Connect the layers together
    def connect(self):
        # From the first layer to the last
        allLayers = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]
        for i, layer in enumerate(allLayers):
            if i + 1 < len(allLayers): # Stop at the layer before the last layer 
                layer.connect_to(allLayers[i + 1])

        self.connected = True


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
        self.connect()


    # Split the input data into 70% training and 30% testing
    def split_data(self):
        print("[*] Splitting input elements")
        # Loop through each class and shuffle the inputs
        for class_, inputs in self.inputs.items():
            # Only keep N elements per class
            inputs = inputs[range(self.samplesPerClass)]

            print("\t-> Selecting training/testing data for class: {}".format(class_))
            random.shuffle(inputs)

            # Take the first 80% elements to use them as training data
            nbSamples = int(round(0.7 * len(inputs)))
            for i in range(0, nbSamples):
                self.trainingData.append({
                    'class': class_,
                    'pixels': inputs[i] / 255 # Normalize values to [0,1]
                })

            # The rest is of course the test data
            for i in range(nbSamples, len(inputs)):
                self.testingData.append({
                    'class': class_,
                    'pixels': inputs[i] / 255 # Normalize values to [0,1]
                })


        print("[*] Shuffling training data")
        random.shuffle(self.trainingData)
        print("[*] Shuffling testing data")
        random.shuffle(self.testingData)
        # Clear the inputs, they aren't need anymore
        del self.inputs


    def back_propagate(self):
        # TODO: Use mini-batches for the gradient descent
        # TODO: Update the biases
        
        # For the output layer:
        for i, neuron in enumerate(list(self.outputLayer.neurons.values())):
            for synapse in neuron.synapses_from:
                errorForOutput = -(self.expectedOutputs[i] - neuron.value) # Gradient of the total error with respect to the output of the neuron
                neuronValForNeuronNet = neuron.value * (1 - neuron.value) # Partial derivative of the activation function
                neuronNetForNeuronWeight = synapse.neuron.value # Gradient of the net input with respect to the weight
                errorSignal = errorForOutput * neuronValForNeuronNet * neuronNetForNeuronWeight
                synapse.updatedWeight = synapse.weight - (self.learningRate * errorSignal)

        # TODO: Fix this because it will most likely only work for one hidden layer...
        # For the hidden layers:
        for layer in self.hiddenLayers:
            for neuron in layer.neurons:
                for synapse in neuron.synapses_from:
                    errorForOutput = 0
                    outputNeurons = list(self.outputLayer.neurons.values())
                    for i in range(self.outputLayer.size):
                        errorForOutput += -(self.expectedOutputs[i] - outputNeurons[i].value) * outputNeurons[i].value * (1 - outputNeurons[i].value) * synapse.weight # Probably the wrong synapse/weight !! We want the weight of the following synpase !

                    neuronValForNeuronNet = neuron.value * (1 - neuron.value) # Partial derivative of the activation function
                    neuronNetForNeuronWeight = synapse.neuron.value # Partial derivative 
                    errorSignal = errorForOutput * neuronValForNeuronNet * neuronNetForNeuronWeight
                    synapse.updatedWeight = synapse.weight - (self.learningRate * errorSignal)

        # Now apply the updated weights !
        for layer in [self.outputLayer] + self.hiddenLayers:
            if type(layer.neurons) is dict: # For the output layer
                neurons = list(layer.neurons.values())
            else: # For the other layers
                neurons = layer.neurons
            
            for neuron in neurons:
                for synapse in neuron.synapses_from:
                    synapse.weight = synapse.updatedWeight
    

    def train(self):
        print("[*] Training neural network...")
        bar = progressbar.ProgressBar(maxval=len(self.trainingData), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for index, element in enumerate(self.trainingData):
            # Setting the input neuronns' value to the pixels' value of the current element
            for i, inputNeuron in enumerate(self.inputLayer.neurons):
                inputNeuron.set_value(element['pixels'][i])
            
            self.expectedOutputs = []
            # Set the expected output layer's outputs' values accordingly
            for classLabel in self.outputLayer.neurons.keys():
                if classLabel == element['class']:
                    self.expectedOutputs.append(1)
                else:
                    self.expectedOutputs.append(-1)

            # Run the neural network for the current input
            for layer in self.hiddenLayers + [self.outputLayer]:
                layer.feed_forward()

            # Calculate the error of this training element for output neurons
            for i, outputNeuron in enumerate(list(self.outputLayer.neurons.values())):
                self.totalError += math.pow((outputNeuron.value - self.expectedOutputs[i]), 2) / 2 # Squarred error
            
            # Adjust weights and biases
            self.back_propagate()
            bar.update(index + 1)


        bar.finish()
        print("[*] Done !")


    def test(self):
        print("[*] Classifying test samples...")
        bar = progressbar.ProgressBar(maxval=len(self.testingData), widgets=[progressbar.Bar('=', '[', ']', ' ', progressbar.Percentage())])
        bar.start()
        successfulGuesses = 0
        for index, element in enumerate(self.testingData):
            # Setting the input neuronns' value to the pixels' value of the current element
            for i, inputNeuron in enumerate(self.inputLayer.neurons):
                inputNeuron.set_value(element['pixels'][i])
            
            # Run the neural network for the current input
            for layer in self.hiddenLayers + [self.outputLayer]:
                layer.feed_forward()

            # Check if it got it right
            if list(self.outputLayer.neurons.keys())[np.argmax([neuron.value for neuron in list(self.outputLayer.neurons.values())])] == element['class']:
                successfulGuesses += 1

            bar.update(index + 1)

        bar.finish()
        print("[*] Done !")
        print("[*] Success rate: {}%".format(round(successfulGuesses / len(self.testingData) * 100)))
            

if __name__ == "__main__":
    random.seed()
    # Using npz files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/
    neuralNetwork = Network(nbPixels = 28*28, samplesPerClass = 500, nbClasses = 4, learningRate = 0.3)
    # neuralNetwork.add_hidden_layer(16)
    # neuralNetwork.add_hidden_layer(16)
    print("[*] Loading data sets")
    neuralNetwork.set_inputs({
        'sword': np.load('datasets/full_numpy_bitmap_sword.npy'),
        'skull': np.load('datasets/full_numpy_bitmap_skull.npy'),
        'skateboard': np.load('datasets/full_numpy_bitmap_skateboard.npy'),
        'pizza': np.load('datasets/full_numpy_bitmap_pizza.npy')
    })
    # For 5 epochs
    for i in range(5):
        neuralNetwork.train()
        neuralNetwork.test()

