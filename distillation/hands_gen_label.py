import tensorflow as tf
import os
import h5py as hdf5
import numpy as np

import neuralnetwork.handetector as hdete

#define a Session
sess = tf.Session()
#load output net
with tf.variable_scope('onet'):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 48, 48, 1])
    onet = hdete.onet({'data': inputs})
    onet.is_training = False
    onet.setup()
onet.load(sess, 'model/onet.pb')
#Set the output of onet as the label

prob = onet.layers['prob']
coor = onet.layers['coor']
label = tf.concat([prob, coor], axis=-1)

#where the hdf5 file are stored
data_dir = '/ssd/rnn_sh/workspace/tools/data'

which_net = 'onet'
imsize = 48
#Initialize a estimator
reg_weights = 0.5

#get the output of the large model
for i in range(1, 6):
    print('We are process %d file...' % i)
    #load the HDF5 file from SSD
    train_file = os.path.join(data_dir, 'train48_hand%d.h5' % (i+1))
    h5file = hdf5.File(train_file)
    dataset_data = h5file['/data']
    data = dataset_data.value
    output = sess.run(label, feed_dict={'input': data})
    h5file.close()
    #save output
    dest_file = os.path.join(data_dir, 'tran48_hand_onetout_%d.h5' % (i + 1))
    h5file = hdf5.File(dest_file)
    h5file.create_dataset(name='label', data=output, dtype='float32')