import h5py as hdf5
import numpy as np


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
h5file.close()
h5file = hdf5.File('/tmp/cifar10_data/cifar10_label_cifaran.h5')
soft_label = h5file['/label'].value
soft_label = np.reshape(soft_label, [50000, 10])
train_label = np.concatenate([train_label, soft_label], axis=1)
print("end")