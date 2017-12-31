import tensorflow as tf
from six import string_types, iteritems
import numpy as np
# import string
import time

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
#Cause there common operation for layers in neural network In-process-Out=In-process-Out...
#So we abstract it as a function
def layers(op):
    def abstract_layer(self, *args, **kwargs):
        #set the name
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        #input of cnn
        if len(self.intermediate) == 0:
            raise ReferenceError("Layer %s does not have input!"%name)
        elif len(self.intermediate) == 1:
            input_nn = self.intermediate[0]
        else:
            input_nn = list(self.intermediate)
        #process
        # before_t = time.time()
        output_nn = op(self, input_nn, *args, **kwargs)
        # after_t = time.time()
        # print("The layer pwgconv %s : time consuming is %f" % (name, after_t - before_t))
        self.layers[name] = output_nn
        self.feed(output_nn)
        return self
    return abstract_layer

class handNN(object):
    def __init__(self, input_nn, trainable=True, is_training=True):
        #Set input of neural network
        self.inputs = input_nn
        #Set whether the network can be trained
        self.is_training = is_training
        #The variable for keep the intermediate results
        self.intermediate = []
        #set layers
        self.layers = dict(input_nn)
        #set the order of channels
        self.channels_axis = -1
        self.trainable = trainable
        #Finally, we initialize the neural network
        # self.setup()

    #Set the setup function, which should be defined by users
    def setup(self):
        raise NotImplementedError("The users should realize the setup() function by yourself...")

    #get the layer name according the layer from same type
    def get_unique_name(self, prefix):
        layer_id = sum(t.startswith(prefix) for t,_ in self.layers.items())
        return '%s_%d'%(prefix, layer_id+1)

    #set the feed function
    def feed(self, *args):
        assert len(args) != 0
        #clear the intermediate
        self.intermediate = []
        #The input maybe two types: string or tensor
        #If it is string, we get the corresponding tensor as layers
        for arg in args:
            if isinstance(arg, string_types):
                try:
                    arg = self.layers[arg]
                except KeyError:
                    raise KeyError("Layer %s not found in the neural network!"%arg)
            #Now it should be a tensor
            self.intermediate.append(arg)
        return self
    #define the function for making variables
    def make_variables(self, name, shape, initializer='TRUNCATED'):
        if initializer.lower() == 'truncated':
            initialization = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.1)
        elif initializer.lower() == 'zeros':
            initialization = tf.zeros(shape=shape)
        elif initializer.lower() == 'gamma':
            initialization = tf.random_gamma(shape=shape, alpha=1.5, beta=2.0)
        else:
            # raise RuntimeWarning('Initialization method %s does not support'%initializer)
            initialization = tf.random_normal(shape=shape, mean=0.0, stddev=0.1)
        return tf.get_variable(name=name, initializer=initialization, trainable=self.trainable)
    #convolutional layer
    @layers
    def conv(self, input_nn, k_h, k_w, s_h, s_w, out_channels, name, padding='VALID', initializer='GAUSSIAN'):
        #We get the depth of last feature map
        in_channels = int(input_nn.get_shape()[-1])
        #We define the convolutional function
        convolue = lambda x, kernel: tf.nn.conv2d(x, kernel, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            #define the weights in convolutional layer
            weight = self.make_variables(name='weight', shape=[k_h, k_w, in_channels, out_channels], initializer=initializer)
            bias = self.make_variables(name='bias', shape=[out_channels])
            output = convolue(input_nn, weight)
            output = tf.add(output, bias)
            return output

    #batch normalization layer
    @layers
    def batch_norm(self, input_nn, name, is_training=True):
        with tf.variable_scope(name):
            output = tf.layers.batch_normalization(inputs=input_nn, axis=3,
                                                   momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                   center=True, scale=True, training=is_training, name=name, reuse=False, fused=True)
            return output

    #Activating function
    @layers
    def activate(self, input_nn, name, activation='ReLU'):
        with tf.variable_scope(name) as scope:
            if activation.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=name)
                return output
            elif activation.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=name)
                return output
            elif activation.lower() == 'prelu':
                i = int(input_nn.get_shape()[-1])
                alpha = self.make_variables('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, tf.subtract(0.0, tf.nn.relu(tf.subtract(0.0, input_nn))))
                # output = tf.nn.relu(input_nn) - alpha*tf.nn.relu(-input_nn)
                return output
            else:
                raise RuntimeError('Unknow activations: %s'%activation)

    #define the FC layer
    @layers
    def fc(self, input_nn, out_channels, name, initializer='GAUSSIAN'):
        #Get the input dimension of this layer
        in_shape = input_nn.get_shape().as_list()
        in_dimension = 1
        for num_dim in in_shape[1:]:
            in_dimension = in_dimension*int(num_dim)
        #add a fully connected layer
        with tf.variable_scope(name):
            weight = self.make_variables('weight', shape=[in_dimension, out_channels], initializer=initializer)
            bias = self.make_variables('bias', shape=[out_channels])
            #before multiple, we flat the matrix into vector
            featmap_flat = tf.reshape(input_nn, [-1, in_dimension])
            output = tf.add(tf.matmul(featmap_flat, weight), bias)
            return output

    #define the pooling layer
    @layers
    def pool(self, input_nn, k_h, k_w, s_h, s_w, name, ptype_nn='MAX', padding='SAME'):
        with tf.variable_scope(name):
            if ptype_nn.lower() == 'max':
                output = tf.nn.max_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            elif ptype_nn.lower() == 'avg':
                output = tf.nn.avg_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            else:
                raise KeyError('Unknow pooling kernel %s'%ptype_nn)

    #define the dropout layer
    @layers
    def dropout(self, input_nn, keep_prob, name):
        with tf.variable_scope(name):
            output = tf.nn.dropout(input_nn, keep_prob=keep_prob)
            return output

    #Define a layer named reshape
    @layers
    def reshape(self, input_nn, shape, name):
        with tf.variable_scope(name):
            output = tf.reshape(input_nn, shape=shape, name=name)
            return output

    @layers
    def softmax(self, target, axis, name=None):
        # max_axis = tf.reduce_max(target, axis, keep_dims=True)
        # target_exp = tf.exp(target-max_axis)
        # normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        # softmax = tf.div(target_exp, normalize, name)
        # return softmax
        softmax = tf.nn.softmax(target, dim=axis, name=name)
        return softmax

    #define the shuffleNet Unit
    @layers
    def shuffle_unit(self, input_nn, group, name, ptype_nn='null', padding='SAME', out_op='ADD'):\
        #The computation quantity is (I*I/g+I*I+9*I)/4, where I is the channels of input feature map
        #get the number of channels of input
        in_dim = int(input_nn.get_shape()[-1])
        assert in_dim % (4*group) == 0
        assert in_dim % 4 == 0
        bot_dim = int(in_dim/4)
        if ptype_nn.lower() == 'null':
            strides_3x3 = [1, 1, 1, 1]
        else:
            strides_3x3 = [1, 2, 2, 1]
        with tf.variable_scope(name):
            short_cut = input_nn
            if ptype_nn.lower() != 'null':
                print('pool_%s' % name)
                short_cut = self.__pool(short_cut, 3, 3, 2, 2, name='pool_%s' % name, ptype_nn=ptype_nn, padding=padding)
            #1x1 point-wise group convolution
            output = self.__pwgconv(input_nn, group, bot_dim, name='pwgc1', shuffle=True, activation='ReLU')
            #3x3 depth-wise convolution
            in_dim_3x3 = int(output.get_shape()[-1])
            kernel = self.make_variables(name='dw_kernel', shape=[3, 3, in_dim_3x3, 1])
            output = tf.nn.depthwise_conv2d(input=output, filter=kernel, strides=strides_3x3, padding=padding)
            output = self.__batch_norm(output, name='bn_%s' % name, is_training=self.is_training)
            #1x1 point-wise group convolution
            output = self.__pwgconv(output, group, in_dim, name='pwgc2', shuffle=False, activation='null')
            if out_op.lower() == 'add':
                output = tf.add(output, short_cut)
            else:
                output = tf.concat([output, short_cut], axis=-1)
            # output = self.activate(output, name='relu', activation='ReLU')

            return output

    def __pwgconv(self, input_nn, group, out_channels, name, shuffle=False, activation='null'):
        # get the number of channels of input
        in_dim = int(input_nn.get_shape()[-1])
        assert in_dim % group == 0
        assert out_channels % group == 0
        group_in_dim = int(in_dim/group)
        group_out_dim = int(out_channels/group)
        convolue_1x1 = lambda x, kernel: tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        with tf.variable_scope(name):
            # define kernels
            weight = []
            output = []
            input_split = tf.split(input_nn, group, axis=-1)
            bias = self.make_variables('bias', [out_channels])
            for i in range(0, group):
                weight.append(self.make_variables('weight%d' % i, [1, 1, group_in_dim, group_out_dim]))
                output_group = convolue_1x1(input_split[i], weight[i])
                output.append(output_group)
            if shuffle:
                output = tf.transpose(output, [1, 2, 3, 4, 0])
            else:
                output = tf.transpose(output, [1, 2, 3, 0, 4])
            fm_h = int(output.get_shape()[1])
            fm_w = int(output.get_shape()[2])
            output = tf.reshape(output, shape=[-1, fm_h, fm_w, out_channels])
            output = tf.add(output, bias)
            output = self.__batch_norm(output, name='bn', is_training=self.is_training)
            if not activation.lower() == 'null':
                output = self.__activate(output, name='relu', activation=activation)
            # get the N H W of output
            return output

    def __batch_norm(self, input_nn, name, is_training=True):
        # if not is_training:
        #     print("NO TRAINING")
        # else:
        #     print("WITH TRAINING")
        # with tf.variable_scope(name):
            # if is_training:
        output = tf.layers.batch_normalization(inputs=input_nn, axis=3,
                                                momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                                center=True, scale=True, training=is_training, fused=True)
            # else:
            #     print("NO BATCH NORMALIZATION")
            #     output = input_nn
        return output

    def __activate(self, input_nn, name, activation='PReLU'):
        with tf.variable_scope(name):
            if activation.lower() == 'relu':
                output = tf.nn.relu(input_nn, name=name)
                return output
            elif activation.lower() == 'sigmoid':
                output = tf.nn.sigmoid(input_nn, name=name)
                return output
            elif activation.lower() == 'prelu':
                i = int(input_nn.get_shape()[-1])
                alpha = self.make_variables('alpha', shape=(i,))
                output = tf.nn.relu(input_nn) + tf.multiply(alpha, tf.subtract(0.0, tf.nn.relu(tf.subtract(0.0, input_nn))))
                return output
            else:
                raise RuntimeError('Unknow activations: %s' % activation)

    def __pool(self, input_nn, k_h, k_w, s_h, s_w, name, ptype_nn='MAX', padding='SAME'):
        with tf.variable_scope(name):
            if ptype_nn.lower() == 'max':
                output = tf.nn.max_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            elif ptype_nn.lower() == 'avg':
                output = tf.nn.avg_pool(input_nn, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
                return output
            else:
                raise KeyError('Unknow pooling kernel %s'%ptype_nn)

    def load(self, sess, in_path):
        #load the model from frozen graph
        with tf.gfile.GFile(in_path, 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        nodes = [node for node in graph_def.node]
        container = [ct for ct in nodes if ct.op == 'Const']
        for ct in container:
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(ct.name)
                sess.run(var.assign(ct.attr['value'].tensor))

    def save(self, sess, out_path='model/model_graph.pb'):
        #save the neural network
        vars = tf.all_variables()
        out_names = [var.name.split(':', 1)[0] for var in vars]
        in_graph_def = sess.graph.as_graph_def()
        out_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                     input_graph_def=in_graph_def,
                                                                     output_node_names=out_names)
        with tf.gfile.GFile(out_path, 'wb') as file:
            file.write(out_graph_def.SerializeToString())
    def freeze_model(self, sess, out_names, out_path='model/model_graph.pb'):
        #save the neural network
        # vars = tf.all_variables()
        # out_names = [var.name.split(':', 1)[0] for var in vars]
        in_graph_def = sess.graph.as_graph_def()
        out_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                     input_graph_def=in_graph_def,
                                                                     output_node_names=out_names)
        with tf.gfile.GFile(out_path, 'wb') as file:
            file.write(out_graph_def.SerializeToString())

    @layers
    def mobile_unit(self, input_nn, filters_num, name, strdies=[1, 1, 1, 1], padding='SAME', width_multiplier=1, alpha=1.0):
        ptw_filters = int(filters_num*alpha)
        in_dim_dwc = int(input_nn.get_shape()[self.channels_axis])
        conv1x1 = lambda x, w: tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        with tf.variable_scope(name):
            kernel = self.make_variables(name='dw_kernel', shape=[3, 3, in_dim_dwc, width_multiplier])
            output = tf.nn.depthwise_conv2d(input=input_nn, filter=kernel, strides=strdies, padding=padding)
            output = self.__batch_norm(output, name='bn1', is_training=self.is_training)
            output = self.__activate(output, name='relu1', activation='ReLU')
            in_dim_pwc = int(output.get_shape()[self.channels_axis])
            weight = self.make_variables(name='weight', shape=[1, 1, in_dim_pwc, ptw_filters])
            output = conv1x1(output, weight)
            output = self.__batch_norm(output, name='bn2', is_training=self.is_training)
            output = self.__activate(output, name='relu2', activation='ReLU')
            return output

    @layers
    def mobile_shortcut(self, input_nn, filters_num, name, strdies=[1, 1, 1, 1], padding='SAME', width_multiplier=1, alpha=1.0):
        ptw_filters = int(filters_num*alpha)
        in_dim_dwc = int(input_nn.get_shape()[self.channels_axis])
        conv1x1 = lambda x, w, stride: tf.nn.conv2d(x, w, strides=stride, padding='SAME')
        short_cut = input_nn
        with tf.variable_scope(name):
            #mobilenet unit
            kernel = self.make_variables(name='dw_kernel', shape=[3, 3, in_dim_dwc, width_multiplier])
            output = tf.nn.depthwise_conv2d(input=input_nn, filter=kernel, strides=strdies, padding=padding)
            output = self.__batch_norm(output, name='bn1', is_training=self.is_training)
            output = self.__activate(output, name='relu1', activation='ReLU')
            in_dim_pwc = int(output.get_shape()[self.channels_axis])
            weight = self.make_variables(name='weight', shape=[1, 1, in_dim_pwc, ptw_filters])
            output = conv1x1(output, weight, stride=[1, 1, 1, 1])
            output = self.__batch_norm(output, name='bn2', is_training=self.is_training)
            #short cut
            kernel_sc = self.make_variables(name='conv1x1', shape=[1, 1, in_dim_dwc, ptw_filters])
            short_cut = conv1x1(short_cut, kernel_sc, stride=strdies)
            short_cut = self.__batch_norm(short_cut, name='bn3', is_training=self.is_training)
            output = output + short_cut
            output = self.__activate(output, name='relu2', activation='ReLU')
            return output