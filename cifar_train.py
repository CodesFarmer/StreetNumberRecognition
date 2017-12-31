import tensorflow as tf
import os
import h5py as hdf5
import numpy as np

import neuralnetwork.handnn as hnn

_WEIGHT_DECAY = 4e-3
_MOMENTUM = 0.9

class cifar_nn(hnn.handNN):
    def setup(self):
        (
            self.feed('data')
                .conv(5, 5, 1, 1, 32, name='conv1', padding='SAME')
                .activate(name='relu1', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool1', padding='VALID')
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
                .conv(3, 3, 1, 1, 16, name='conv1', padding='SAME')
                .activate(name='relu1', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool1', padding='VALID')
                .conv(3, 3, 1, 1, 16, name='conv2', padding='SAME')
                .activate(name='relu2', activation='ReLU')
                .pool(3, 3, 2, 2, name='pool2', ptype_nn='AVG', padding='VALID')
                .conv(3, 3, 1, 1, 32, name='conv4', padding='SAME')
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

def model_fn(features, labels, mode):
    inputs = tf.reshape(features['x'], [-1, 32, 32, 3])
    inputs = tf.cast(inputs, tf.float32)
    with tf.variable_scope('cifar_q'):
        neuralnetwork = cifar_nn_q({'data': inputs})
    neuralnetwork.is_training = mode == tf.estimator.ModeKeys.TRAIN
    neuralnetwork.setup()
    logits = neuralnetwork.layers['fc2']
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)
        global_step = tf.train.get_or_create_global_step()
        #set the learning rate
        initial_learning_rate = 0.01
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(500 * epoch) for epoch in [10, 20, 30]]
        values = [initial_learning_rate * decay for decay in [1, 0.5, 0.1, 0.05]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        #For comparision, we set the learning fixed
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    prediction = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1),
        tf.argmax(logits, axis=1)
    )
    metrics = {'accuracy': accuracy}
    #set the accuracy for plot figure
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        train_op=train_op,
        loss=loss,
        predictions=prediction,
        eval_metric_ops=metrics
    )


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
cifar_classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='model', config=run_config)
train_epochs = 40
h5file = hdf5.File('/tmp/cifar10_data/CIFAR-10.hdf5')
train_data = h5file['/normalized_full/training/default']
train_data = train_data.value[0, :, :, :, :]
tr_label = h5file['/normalized_full/training/targets']
train_label = np.zeros(shape=[tr_label.shape[1], 10])
for tl in range(50000):
    train_label[tl, tr_label.value[0, tl, 0]] = 1
test_data = h5file['/normalized_full/test/default']
test_data = test_data.value[0, :, :, :, :]
te_label = h5file['/normalized_full/test/targets']
test_label = np.zeros(shape=[te_label.shape[1], 10])
for tl in range(10000):
    test_label[tl, te_label.value[0, tl, 0]] = 1
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array(train_data)},
    y=np.array(train_label),
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array(test_data)},
    y=np.array(test_label),
    batch_size=2000,
    num_epochs=1,
    shuffle=True
)
for _ in range(train_epochs // 1):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    cifar_classifier.train(
        input_fn=train_input_fn,
        steps=500)

    # Evaluate the model and print results
    eval_results = cifar_classifier.evaluate(
        input_fn=test_input_fn)
    print(eval_results)