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
import progressbar
import multiprocessing
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
    testData = []
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

            print("\t-> Selecting training/test data for class: {}".format(class_))
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
                self.testData.append({
                    'class': class_,
                    'pixels': inputs[i] / 255 # Normalize values to [0,1]
                })


        print("[*] Shuffling training data")
        random.shuffle(self.trainingData)
        print("[*] Shuffling test data")
        random.shuffle(self.testData)
        # Clear the inputs, they aren't need anymore
        del self.inputs


    # Stochastic back propagation
    def back_propagate(self):
        # TODO: Use mini-batches (of size > 1) for the gradient descent (is it gonna improve the computation speed or just the training speed ?...)
        # TODO: Update the biases
        # TODO: Threads !
        
        # For the output layer:
        for i, neuron in enumerate(list(self.outputLayer.neurons.values())):
            for synapse in neuron.synapses_from:
                errorForOutput = -(self.expectedOutputs[i] - neuron.value) # Gradient of the total error with respect to the output of the neuron
                neuronValForNeuronNet = neuron.value * (1 - neuron.value) # Partial derivative of the activation function
                neuronNetForNeuronWeight = synapse.neuron.value # Gradient of the net input with respect to the weight
                errorSignal = errorForOutput * neuronValForNeuronNet * neuronNetForNeuronWeight
                synapse.updatedWeight = synapse.weight - (self.learningRate * errorSignal)

        # TODO: Fix this because it will most likely only work for one hidden layer... But maybe not...
        # For the hidden layers:
        for layer in self.hiddenLayers:
            for neuron in layer.neurons:
                for synapse in neuron.synapses_from:
                    errorForOutput = 0
                    outputNeurons = list(self.outputLayer.neurons.values())
                    for i in range(self.outputLayer.size):
                        errorForOutput += -(self.expectedOutputs[i] - outputNeurons[i].value) * outputNeurons[i].value * (1 - outputNeurons[i].value) * neuron.synapses_to[i].weight 

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
        pool = multiprocessing.Pool(processes=4)
        bar = progressbar.ProgressBar(maxval=len(self.trainingData), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for index, element in enumerate(self.trainingData):
            # Setting the input neuronns' value to the pixels' value of the current element
            for i, inputNeuron in enumerate(self.inputLayer.neurons):
                inputNeuron.value = element['pixels'][i]
            
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
                self.totalError += ((outputNeuron.value - self.expectedOutputs[i]) ** 2) / 2 # Squarred error
            
            # Adjust weights and biases
            self.back_propagate()

            bar.update(index + 1)


        bar.finish()
        print("[*] Done !")


    def test(self):
        print("[*] Classifying test samples...")
        bar = progressbar.ProgressBar(maxval=len(self.testData), widgets=[progressbar.Bar('=', '[', ']', ' ', progressbar.Percentage())])
        bar.start()
        successfulGuesses = 0
        for index, element in enumerate(self.testData):
            # Setting the input neurons' value to the pixels' value of the current element
            for i, inputNeuron in enumerate(self.inputLayer.neurons):
                inputNeuron.value = element['pixels'][i]
            
            # Run the neural network for the current input
            for layer in self.hiddenLayers + [self.outputLayer]:
                layer.feed_forward()

            # Check if it got it right
            if list(self.outputLayer.neurons.keys())[np.argmax([neuron.value for neuron in list(self.outputLayer.neurons.values())])] == element['class']:
                successfulGuesses += 1

            bar.update(index + 1)

        bar.finish()
        print("[*] Done !")
        print("[*] Success rate: {}%".format(round(successfulGuesses / len(self.testData) * 100)))
            

if __name__ == "__main__":
    random.seed()
    # Using npz files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/

    # Best configuration so far:
    # neuralNetwork = Network(nbPixels = 28*28, samplesPerClass = 20000, nbClasses = 3, learningRate = 0.35)
    # neuralNetwork.add_hidden_layer(64)
    # neuralNetwork.add_hidden_layer(16)
    neuralNetwork = Network(nbPixels = 28*28, samplesPerClass = 350, nbClasses = 3, learningRate = 0.5)
    neuralNetwork.add_hidden_layer(64)
    neuralNetwork.add_hidden_layer(16)
    print("[*] Creating Neural Network: samplesPerClass = {}, learningRate = {}, hiddenLayers: {}".format(neuralNetwork.samplesPerClass, neuralNetwork.learningRate, [hl.size for hl in neuralNetwork.hiddenLayers]))
    print("[*] Loading data sets")
    neuralNetwork.set_inputs({
        'sword': np.load('datasets/full_numpy_bitmap_sword.npy'),
        'skull': np.load('datasets/full_numpy_bitmap_skull.npy'),
        'skateboard': np.load('datasets/full_numpy_bitmap_skateboard.npy'),
        # 'pizza': np.load('datasets/full_numpy_bitmap_pizza.npy')
    })
    # For 5 epochs
    for i in range(5):
        neuralNetwork.train()
        neuralNetwork.test()

