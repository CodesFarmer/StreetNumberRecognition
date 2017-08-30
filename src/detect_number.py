'''
This file will detect the number in the pictures
'''

import tensorflow as tf

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        #set the layer name
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        #Set the layer's input
        if len(self.terminals == 1):
            layer_input = self.terminals[0]
        elif len(self.terminals == 0):
            raise RuntimeError("There does not exist input for layer %s"%name)
        else:
            layer_input = list(self.terminals)
        #Get the output through the layer
        layer_output = op(self, layer_input, *args, **kwargs)
        #Set the layer
        self.layers[name] = layer_output


class Network(object):
    def __init__(self, inputs, labels, trainable=True):
        self.inputs = inputs
        self.labels = labels
        self.terminals = []
        #get the layer name
        self.layers = dict(inputs)
        #set the mark indicate that the network can be trained
        self.trainable = trainable
        #set up the structure of network
        self.setup()
    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass.')
    def get_unique_name(self, prefix):
        ident = sum(t.startwith(prefix) for t,_ in self.layers.item()) + 1
        return '%s_%d'%(prefix, ident)
    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding='SAME', group=1, biased=True):
        #verify the input padding is existing
        assert padding in ('SAME', 'VALID')
        c_i = int(input.getshape()[-1])
        #where c_i is the channels of input feature map, and c_o is about output
        assert c_i%group == 0
        assert c_o%group == 0
        #first, we will generate a convolutional layer
        convolve = lambda i,k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        #set the input and kernel for the convolutional layer
        with tf.variable_scope(name) as scope:
            kernel = tf.