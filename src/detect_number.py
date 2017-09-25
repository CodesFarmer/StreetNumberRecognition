'''
This file will detect the number in the pictures
'''

import tensorflow as tf
from six import string_types, iteritems

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
    def __init__(self, inputs, trainable=True):
        self.inputs = inputs
        # self.labels = labels
        self.terminals = []
        #get the layer name
        self.layers = dict(inputs)
        #set the mark indicate that the network can be trained
        self.trainable = trainable
        #set up the structure of network
        self.setup()
    def setup(self):
        raise NotImplementedError('Must be implemented by the subclass.')
    def get_output(self):
        return self.terminals[-1]
    def get_unique_name(self, prefix):
        ident = sum(t.startwith(prefix) for t,_ in self.layers.item()) + 1
        return '%s_%d'%(prefix, ident)
    def make_var(self, name, shape):
        "create a new tensorflow variable"
        initials = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial_value=initials, name=name)
        # return tf.get_variable(name=name,validate_shape=shape, trainable=self.trainable)
    def feed(self, *args):
        assert len(args) != 0
        self.terminals = []
        for feed_layers in args:
            if isinstance(feed_layers, string_types):
                try:
                    feed_layers = self.layers[feed_layers]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % feed_layers)
            self.terminals.append(feed_layers)
        return self
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
            kernel = self.make_var('weights', [k_h, k_w, c_i//group, c_o])
            output = convolve(input, kernel)
            if biased:
                biases = self.make_var('biases', [1,1,1,c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)
        return output

    @layer
    def pooling(self, input, k_h, k_w, s_h, s_w, name, padding='SAME'):
        assert padding in ('SAME', 'VALID')
        output = tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1],
                                strides=[1, s_h, s_w, 1],
                                padding=padding, name=name)
        return output
    @layer
    def fclayer(self, input, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = input.getshape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', shape=[-1, num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            output = op(feed_in, weights, biases, name=name)
            return output
    @layer
    def softmax(self, target, axis, name=None):
        "softmax function for normalization"
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        output = tf.div(target_exp, normalize, name=name)
        return output
    @layer
    def prelu(self, input, name):
        with tf.variable_scope(name) as scope:
            i = int(input.getshape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(input) + tf.multiply(alpha, -tf.nn.relu(-input))
        return output

class PNet(Network):
    def setup(self):
        (self.feed('data')
            .conv(3,2,10,2,1,padding='VALID', relu=False, name='conv1'))
            # .prelu(name='PRelu1')
            # .conv(3,3,16,1,1,padding='VALID', relu=False, name='conv2')
            # .prelu(name='PRelu2')
            # .conv(3,3,32,1,1,padding='VALID', relu=False, name='conv3')
            # .prelu(name='PRelu3')
            # .conv(1,1,2,1,1,padding='VALID', relu=False, name='conv4-1')
            # .softmax(3,name='prob1'))

        # (self.feed('PRelu3')
        #     .conv(1,1,4,1,1,padding='VALID', relu=False, name='conv4-2'))

def CreateMTCNN(sess):
    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet = PNet({'data': data})
        sess.run(tf.global_variables_initializer())
    pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
    return pnet_fun

def Network_loss(labels, prediction):
    error_depth = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=prediction)
    cross_entropy = tf.reduce_mean(error_depth)
    return cross_entropy
def Network_train(the_loss, lr):
    optimizer = tf.train.Optimizer(lr)
    trainer = optimizer.minimize(the_loss)
    return trainer
def Train_PNet(sess, input, label, pnet):
    prediction = pnet(input)
    entropy_loss = Network_loss(labels=label, prediction=prediction)
    net_trainer = Network_train(entropy_loss, 0.01)
    sess.run(net_trainer)
