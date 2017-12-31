import tensorflow as tf
import os
import h5py as hdf
import numpy as np

import neuralnetwork.handetector_distilling as hdete

#where the hdf5 file are stored
data_dir = '/home/shenhui/nfs/rnn_sh/workspace/tools/data'

which_net = 'onet'
imsize = 48
batch_size = 128
epochs = 15
iterations_per_epoch = 3000

#Initialize a estimator
reg_weights = 0.5
model_param = {'neuralnetwork': which_net, 'reg_weights': reg_weights, 'epochs': epochs*5, 'max_steps': iterations_per_epoch}
run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
pnet_nn = tf.estimator.Estimator(
    model_fn=hdete.model_nn_fn,
    params=model_param,
    model_dir='model',
    config=run_config
)

#load test data at once
test_file = os.path.join(data_dir, 'test%d_hand.h5' % imsize)
h5file = hdf.File(test_file)
test_label = h5file['/label'].value
test_data = h5file['/data'].value
h5file.close()
test_file = os.path.join(data_dir, 'test48_hand_onetout.h5')
h5file = hdf.File(test_file)
test_soft_label = h5file['/label'].value
h5file.close()
test_label = np.concatenate([test_label, test_soft_label], axis=1)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x':test_data},
    y=test_label,
    batch_size=1024,
    num_epochs=1,
    shuffle=False
)
#cause we have file training hdf5 file, so we load the seperately
train_epochs = epochs
for i in range(0, train_epochs*5):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    #read the data and label from h5 file
    train_file = os.path.join(data_dir, 'train%d_hand_%d.h5' % (imsize, i % 5 + 1))
    h5file = hdf.File(train_file)
    dataset_label = h5file['/label']
    dataset_data = h5file['/data']
    train_label = dataset_label.value
    train_data = dataset_data.value
    h5file.close()
    train_file = os.path.join(data_dir, 'tran%d_hand_onetout_%d.h5' % (imsize, i % 5 + 1))
    h5file = hdf.File(train_file)
    train_soft_label = h5file['/label'].value
    h5file.close()
    train_label = np.concatenate([train_label, train_soft_label], axis=1)
    #shuffle the data
    index = np.random.permutation(train_label.shape[0])
    train_label = train_label[index, :]
    train_data = train_data[index, :]
    #set the input_fn
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_label,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False
    )
    pnet_nn.train(
        input_fn=train_input_fn,
        steps=iterations_per_epoch
    )
    eva_results = pnet_nn.evaluate(
        input_fn=test_input_fn
    )
    print(eva_results)