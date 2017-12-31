import tensorflow as tf
import numpy as np
import h5py as hdf5

_NUMCLASS = 3
def loss_softtarget(labels, logits, temperature=4.0):
    logits = tf.convert_to_tensor(logits)
    labels = tf.convert_to_tensor(labels)
    pre_prob = tf.exp(logits/temperature)
    grt_prob = tf.exp(labels/temperature)
    pre_prob_sum = tf.reduce_sum(pre_prob, axis=-1)
    grt_prob_sum = tf.reduce_sum(grt_prob, axis=-1)
    pre_prob_sum_ = []
    for i in range(0, _NUMCLASS):
        pre_prob_sum_.append(pre_prob_sum)
    grt_prob_sum_ = []
    for i in range(0, _NUMCLASS):
        grt_prob_sum_.append(grt_prob_sum)
    # pre_prob_sum = [pre_prob_sum, pre_prob_sum, pre_prob_sum]
    pre_prob_sum_ = tf.transpose(pre_prob_sum_, [1, 0])
    # grt_prob_sum = [grt_prob_sum, grt_prob_sum, grt_prob_sum]
    grt_prob_sum_ = tf.transpose(grt_prob_sum_, [1, 0])
    pre_prob = tf.div(pre_prob, pre_prob_sum_)
    grt_prob = tf.div(grt_prob, grt_prob_sum_)
    return tf.squared_difference(x=pre_prob, y=grt_prob)

def loss_l2norm(labels, logits, temperature=4):
    logits = tf.convert_to_tensor(logits)
    labels = tf.convert_to_tensor(labels)
    sum_logit = tf.sqrt(tf.reduce_sum(tf.multiply(logits, logits)))
    sum_labels = tf.sqrt(tf.reduce_sum(tf.multiply(labels, labels)))

def cross_entropy_loss(labels, logits):
    labels = tf.convert_to_tensor(labels)
    logits = tf.convert_to_tensor(logits)
    loss_elementwise = tf.multiply(labels, tf.log(logits)) + tf.multiply(1-labels, tf.log(1-logits))
    loss = tf.reduce_sum(loss_elementwise, axis=-1)
    return  loss

sess = tf.Session()
label = [[3.0510e-032, 3.0510e-030, 3.0530e-015], [0.2, 0.2, 0.6]]
logit = [[3.0520e-032, 3.0520e-030, 3.0510e-015], [0.01, 0.01, 0.98]]

# print(sess.run(loss_l2norm(labels=label, logits=logit)))
_T = 2
labels = tf.convert_to_tensor(label)
logits = tf.convert_to_tensor(logit)
labels = tf.divide(labels, _T)
logits = tf.divide(logits, _T)
labels = tf.nn.softmax(logits=labels)
logits = tf.nn.softmax(logits=logits)
# loss_elementwise = tf.multiply(labels, tf.log(logits)) + tf.multiply(1-labels, tf.log(1-logits))
# loss = tf.reduce_sum(loss_elementwise, axis=-1)
loss = _T*_T*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

print(sess.run(labels))
print("=================")
print(sess.run(logits))
print("=================")
print(sess.run(loss))
print(sess.run(tf.reduce_mean(loss)))
#
# h5file = hdf5.File('/tmp/cifar10_data/CIFAR-10.hdf5')
# tr_label = h5file['/normalized_full/training/targets']
# train_label = np.zeros(shape=[tr_label.shape[1], 10])
# for tl in range(50000):
#     train_label[tl, tr_label.value[0, tl, 0]] = 1
# h5file.close()
# h5file = hdf5.File('/tmp/cifar10_data/cifar10_label_cifares_T4.h5')
# soft_label = h5file['/label'].value
# soft_label = np.reshape(soft_label, [50000, 10])
# train_label = np.concatenate([train_label, soft_label], axis=1)