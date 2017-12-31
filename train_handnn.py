import tensorflow as tf
import neuralnetwork.handnn as hnn
import h5py
import numpy as np

_WEIGHT_DECAY = 5e-5
_MOMENTUM = 0.9
class pnet(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
            .conv(3, 3, 1, 1, 10, name='conv1', padding='VALID')
            .activate(name='prelu1', activation='PReLU')
            .pool(2, 2, 2, 2, name='pool1', ptype_nn='MAX', padding='VALID')
            .conv(3, 3, 1, 1, 16, name='conv2', padding='VALID')
            .activate(name='prelu2', activation='PReLU')
            .conv(3, 3, 1, 1, 32, name='conv3', padding='VALID')
            .activate(name='prelu3', activation='PReLU')
            .conv(1, 1, 1, 1, 2, name='prob', padding='VALID')
            .softmax(3, name='prob_sm')
        )
        (   self.feed('prelu3')
            .conv(1, 1, 1, 1, 4, name='coor', padding='VALID')
        )

# in_size = 12
labels = tf.placeholder(dtype=tf.float32, shape=[None, 5])
data = tf.placeholder(dtype=tf.float32, shape=[None, 1, 12, 12])
category, groundtruth = tf.split(labels, [1, 4], 1)
#transpose the data from NCHW to NHWC
inputs = tf.transpose(data, [0, 2, 3, 1])
#there are two part of output: probability and regression
neuralnetwork = pnet({'data': inputs})
reg_weights = 0.5
prob = neuralnetwork.layers['prob']
coor = neuralnetwork.layers['coor']
#define the loss, but we transfer the single label into matrix at first
num_one = tf.constant(-1, tf.int32)
num_zero = tf.constant(0, tf.int32)
category = tf.cast(category, tf.int32)
mask_cat = tf.not_equal(category, num_one)
mask_reg = tf.not_equal(category, num_zero)
category = tf.one_hot(category, 3)
category = category[:, :, 0:2]
category = tf.reshape(category, shape=[-1, 1, 1, 2])
groundtruth = tf.reshape(groundtruth, shape=[-1, 1, 1, 4])
prob = tf.boolean_mask(prob, mask_cat)
category = tf.boolean_mask(category, mask_cat)
coor = tf.boolean_mask(coor, mask_reg)
groundtruth = tf.boolean_mask(groundtruth, mask_reg)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=category, logits=prob))
squared_error = tf.reduce_mean(tf.squared_difference(x=coor, y=groundtruth))
weight_decay = _WEIGHT_DECAY*tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
loss = cross_entropy + reg_weights*squared_error + weight_decay
# loss = cross_entropy
#we set the loss for display
tf.identity(cross_entropy, 'cross_entropy')
tf.summary.scalar('cross_entropy', cross_entropy)
tf.identity(squared_error, 'squared_error')
tf.summary.scalar('squared_error', squared_error)
#set the train_op
learning_rate = 0.001
#draw a curve for display learning rate
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
predictions = {
    'classes': tf.argmax(prob, axis=2),
    'probabilities': tf.nn.softmax(prob, name='softmax_tensor')
}

accuracy = tf.metrics.accuracy(
    tf.argmax(prob, axis=2),
    tf.argmax(category, axis=2)
)

corrections = tf.equal(tf.argmax(category, 2), tf.argmax(prob, 2))
accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


for i in range(1, 20):
    #Load the data
    h5file =  h5py.File('/ssd/rnn_sh/workspace/tools/data/test12_hand.h5', 'r+')
    dataset_label = h5file['/label'].value
    dataset_data = h5file['/data'].value
    index = np.random.permutation(dataset_label.shape[0])
    dataset_label = dataset_label[index, :]
    dataset_data = dataset_data[index, :]
    batch_size = 128
    for j in range(0, 4000):
        input_x = dataset_data[j*128:((j+1)*128-1), :, :, :]
        input_y = dataset_label[j*128:((j+1)*128-1), :]
        sess.run(train_op, feed_dict={data: input_x, labels: input_y})
        if j%500 == 0:
            print('(%d-%d)training_loss: %f, accuracy is %f'%(i, j, sess.run((cross_entropy), feed_dict={data: input_x, labels: input_y}),
                                                              sess.run((accuracy), feed_dict={data: input_x, labels: input_y})))
            # print('The output of prob layer is %r'%(sess.run(prob, feed_dict={data: input_x, labels: input_y})))
            # print(sess.run(category, feed_dict={data: input_x, labels: input_y}))
            # print(sess.run(prob, feed_dict={data: input_x, labels: input_y}))
    saver.save(sess, 'model/pnet_model_%d.ckpt'%i)