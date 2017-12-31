import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.python.client import timeline
import h5py as hdf5
import numpy as np
import time

import neuralnetwork.handnn as hnn

_WEIGHT_DECAY = 4e-3
_MOMENTUM = 0.9

class cifar_nn(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(5, 5, 1, 1, 32, name='conv1', padding='SAME')
            .pool(3, 3, 2, 2, name='pool1', padding='VALID')
            .activate(name='relu1', activation='ReLU')
            .conv(5, 5, 1, 1, 32, name='conv2', padding='SAME')
            .activate(name='relu2', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool2', ptype_nn='AVG', padding='VALID')
            .conv(5, 5, 1, 1, 64, name='conv4', padding='SAME')
            .activate(name='relu3', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool3', ptype_nn='AVG', padding='VALID')
            .fc(64, name='fc1')
            .fc(10, name='fc2')
        )
class cifar_nn_q(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
                .conv(3, 3, 1, 1, 8, name='conv1', padding='SAME')
                .activate(name='relu1', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool1', padding='VALID')
                .conv(3, 3, 1, 1, 8, name='conv2', padding='SAME')
                .activate(name='relu2', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool2', ptype_nn='AVG', padding='VALID')
                .conv(3, 3, 1, 1, 16, name='conv4', padding='SAME')
                .activate(name='relu3', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool3', ptype_nn='AVG', padding='VALID')
                .fc(32, name='fc1')
                .fc(10, name='fc2')
        )


class cifar_shuffle(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 32, name='conv1', padding='SAME')
            .activate(name='relu1', activation='ReLU')
            .pool(3, 3, 2, 2, name='pool1', padding='VALID')
            .shuffle_unit(4, name='shu1', ptype_nn='AVG', padding='VALID', out_op='CONCAT')
            .activate(name='relu2', activation='ReLU')
            # .shuffle_unit(4, name='shu3', ptype_nn='null', padding='SAME', out_op='ADD')
            # .activate(name='relu4', activation='ReLU')
            # .shuffle_unit(4, name='shu4', ptype_nn='null', padding='SAME', out_op='ADD')
            # .activate(name='relu5', activation='ReLU')
            .shuffle_unit(4, name='shu5', ptype_nn='AVG', padding='VALID', out_op='ADD')
            .activate(name='relu6', activation='ReLU')
            .fc(64, name='fc1')
            .fc(10, name='fc2')
        )

class cifar_mobile(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
                .conv(3, 3, 1, 1, 32, name='conv1', padding='SAME')
                .activate(name='relu1', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool1', padding='VALID')
                # .mobile_unit(filters_num=64, name='mbn1', strdies=[1, 1, 1, 1])
                # .mobile_unit(filters_num=64, name='mbn2', strdies=[1, 1, 1, 1])
                .mobile_unit(filters_num=64, name='mbn3', strdies=[1, 2, 2, 1], padding='SAME')
                .mobile_unit(filters_num=64, name='mbn4', strdies=[1, 2, 2, 1], padding='SAME')
                .fc(64, name='fc1')
                .fc(10, name='fc2')
        )


x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
inputs = tf.reshape(x, [-1, 32, 32, 3])
inputs = tf.cast(inputs, tf.float32)
# neuralnetwork = cifar_shuffle({'data' : inputs})
neuralnetwork = cifar_nn({'data' : inputs})
# neuralnetwork = cifar_nn_q({'data' : inputs})
# neuralnetwork = cifar_mobile({'data': inputs})
neuralnetwork.is_training = False
neuralnetwork.setup()
logits = neuralnetwork.layers['fc2']
saver = tf.train.Saver()
sess = tf.Session()

# saver.restore(sess, "model_pool_shfn/model.ckpt-30000")
# saver.restore(sess, "model_sn/model.ckpt-20000")
# saver.restore(sess, "model/model_an/model.ckpt-20000")
# saver.restore(sess, "model/model_mn/model.ckpt-20000")
# saver.restore(sess, "model/model.ckpt-20000")
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))

# print(sess.graph_def)
# neuralnetwork.save(sess, 'model/model_cfan_q.pb')
neuralnetwork.load(sess, 'model_teacher/model_cfan.pb')



h5file = hdf5.File('/tmp/cifar10_data/CIFAR-10.hdf5')
# test_data = h5file['/normalized_full/test/default']
# test_data = test_data.value[0, :, :, :, :]
# te_label = h5file['/normalized_full/test/targets']
# test_label = np.zeros(shape=[te_label.shape[1], 10])
# for tl in range(10000):
#     test_label[tl, te_label.value[0, tl, 0]] = 1
#
# #statistic time consuming
# batch_size = 10000
# batch_num = int(10000/batch_size)
# for i in range(0, batch_num):
#     before_t = time.time()
#     for j in range(0, 10):
#         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#         run_metadata = tf.RunMetadata()
#         accuracy_ = sess.run(accuracy, options=run_options, run_metadata=run_metadata,
#                                feed_dict={x: test_data[i*batch_size:(i+1)*batch_size, :, :, :],
#                                           y: test_label[i*batch_size:(i+1)*batch_size, :]})
#         tl = timeline.Timeline(run_metadata.step_stats)
#         ctf = tl.generate_chrome_trace_format()
#         with open('mobilenet_1_timeline_%d.json'%j, 'w') as file:
#             file.write(ctf)
#         print("The accuracy is %f" % accuracy_)
#     after_t = time.time()
#     print("The time consuming is %f" % ((after_t - before_t)/10.0))


train_data = h5file['/normalized_full/training/default']
train_data = train_data.value[0, :, :, :, :]
tr_label = h5file['/normalized_full/training/targets']
train_label = np.zeros(shape=[tr_label.shape[1], 10])
for tl in range(50000):
    train_label[tl, tr_label.value[0, tl, 0]] = 1
prob_per_axis = neuralnetwork.layers['fc2']
label = []
for iter in range(0, 5):
    label.append(sess.run(prob_per_axis, feed_dict={x: train_data[iter*10000:(iter+1)*10000, :, :, :]}))
    print(sess.run(accuracy, feed_dict={x: train_data[iter*10000:(iter+1)*10000, :, :, :],
                                               y: train_label[iter*10000:(iter+1)*10000, :]}))

f = hdf5.File('/tmp/cifar10_data/cifar10_label_cifaran.h5', 'w')
f.create_dataset(name='label', data=label, dtype='float32')
f.close()