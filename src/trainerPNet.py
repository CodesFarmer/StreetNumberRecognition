'''
Prepare the training data
For test, we only input a single sample
'''
import cv2 as cv
import tensorflow as tf
import os
import detect_number

data_path='../data/'
image = cv.imread(os.path.join(data_path,'12/positive/1.jpg'))
label = [1, 0.06, 0.04, -0.06, -0.18]
sess = tf.Session()
pnet = detect_number.CreateMTCNN(sess)