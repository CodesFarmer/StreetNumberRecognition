'''
This file will detect the number in the pictures
'''

import tensorflow as tf


class PNet:
    def __init__(self, inputs, labels, trainable=True):
        self.inputs = inputs
        self.labels = labels
        #get the layer name
        self.layers = dict(inputs)
        #set the mark indicate that the network can be trained
        self.trainable = trainable
        #set up the structure of network
        self.setup()
    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass.')
    def conv(self):
        print(1)