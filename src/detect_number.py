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
    def make_var(self, name, shape):
        #return the variables
        return tf.get_variable(name, shape, trainable=self.trainable)
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
            #scope is the scopes of variables
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i/group, c_o])
            output = convolve(input, kernel)
            #add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output + biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
            return output
    @layer
    def pooling(self, input, k_h, w_h, s_h, s_w, name, padding='SAME'):
        assert padding in ('SAME', 'VALID')
        return tf.nn.max_pool(input, ksize=[1, k_h, w_h, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)