
'''
import tensorflow as tf
hello = tf.constant("Hello TensorFlow!")
sess = tf.Session()
print(sess.run(hello))

a = "[1,2,3,4]"
print(a[1])

print(a)

for i in range(10):
    print(i)
'''

import cv2
import numpy as np
import tensorflow as tf

# img = cv2.imread("134212.jpg")
# print(img.shape)
# im = img[1:400, 1:1000, :]
# cv2.imshow("test", im)
# cv2.waitKey(0)
#
# print(3**2)
#
# # for i in range(0, 100, 10):
# #     print(i)
# #
#
# A = tf.zeros([1, 12, 12, 3], dtype=tf.float32)
# B = tf.zeros([1, 12, 12, 3], dtype=tf.float32)
# shape_A = A.get_shape()
# print('The shape of A is (%d, %d, %d, %d)'%(shape_A[0], shape_A[1], shape_A[2], shape_A[3]))
# print('The A is %r'%A)
# C = []
# if C == []:
#     print('C is empty')
# C = tf.zeros([1, 12, 12, 3], dtype=tf.float32)
# if C == []:
#     print('C is empty')
#
# if A.get_shape() == B.get_shape():
#     print('The shape of A and B are equal')
# else:
#     print('The shape of A and B are unequal')
# # C = tf.add(C, A)
# # C = tf.add(C, B)
# # C = A + B

# strides = (2, 2)
# # strides = (1,) + strides + (1,)
# strides = (1, 1) + strides
# print(strides)

# label = [0, 1, -1]
# label_vec = tf.one_hot(label, depth=3)
# print(tf.Session.run([label_vec]))

label = [1,2,3,4,5,6,7,8,9,10]
p = [3, 6, 2]
print(label[0:3])

# a = [[[1,1,1]],[[2,2,2]],[[3,3,3]],[[4,4,4]]]
# print(a)
# tensor_a = tf.convert_to_tensor(a, tf.float32)
# tensor_a = tf.reshape(tensor_a, shape=[-1, 1, 1, 3])
# print(tf.Session().run(tensor_a))
#
# A = [[[[-0.18924966, -0.0899481 ]]]]
# A = np.array(A)
# print(A.shape)
# print(A)

var = []
output = []
kernel = []
convolue_1x1 = lambda x, kernel: tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
for i in range(0, 5):
    constant = tf.constant(i+1, dtype=tf.float32)
    var.append(tf.get_variable(name='input_%d' % i, shape=[10, 2, 2, 5], dtype=tf.float32, initializer=tf.ones_initializer))
    var[i] = tf.multiply(var[i], constant)
    kernel.append(tf.get_variable(name='weight_%d' % i, shape=[1, 1, 5, 3], dtype=tf.float32, initializer=tf.ones_initializer))
    output.append(convolue_1x1(var[i], kernel[i]))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
output = tf.transpose(output, [1,2,3,4,0])
fm_h = int(output.get_shape()[1])
fm_w = int(output.get_shape()[2])
output = tf.reshape(output, shape=[-1, fm_h, fm_w, 3*5])
split = tf.split(output, 5, axis=-1)
print(output[0])
print(output[1])
print(sess.run(output[0, 0, 0, :]))
print(sess.run(split[1][0, 0, 0, :]))
# # output = tf.transpose(output, [1,2,3,0,4])
# # output = tf.reshape(output, shape=[-1, 2, 2, 3*5])
# # print(sess.run(output[0, 0, 0, :]))
# saver = tf.train.Saver({'pnet': kernel[0]})
# saver.save(sess, 'model/test_pref.ckpt')

size = output.get_shape().as_list()
print(output.get_shape())

A = []
print(tf.convert_to_tensor(A).get_shape()[0])