import tensorflow as tf
import numpy as np
import cv2
import os
import sys

import neuralnetwork.handetector as hd
if len(sys.argv) == 1:
    raise ("Please the part id of streams...")

part_id = int(sys.argv[1])

root_dir = '/home/shenhui/program/python/gathersamples/handetect_tf'
minsize = 12
factor = 0.709 # scale factor
model_path = []
model_path.append('%s/model/pnet_relu.pb' % root_dir)
model_path.append('%s/model/rnet_relu.pb' % root_dir)
model_path.append('%s/model/onet_mn_relu.pb' % root_dir)
threshold = [0.7, 0.6, 0.6]

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
sess = tf.Session(config=config)
params = []
proposal_pnet, refine_rnet, output_onet = hd.load_dnn(sess, model_path, params)

#load the file lists
with open('data/streams_%d.txt' % part_id, 'r') as file:
    content = file.readlines()
num_lines = len(content)

line_stride = 2

for iter in range(24100, num_lines):
    if (iter-(1+line_stride)) % 100 == 0:
        print("%d/%d" % (iter-3, num_lines-(1+line_stride)))

    cur_img = content[iter].rstrip()
    xml_name = cur_img
    xml_name = xml_name.replace('png', 'xml')
    xml_name = xml_name.replace('jpg', 'xml')
    xml_name = xml_name.replace('cam0', 'xml')
    if not os.path.exists(xml_name):
        continue
    frame_1 = content[iter - 1 - line_stride].rstrip()
    frame_2 = content[iter - 1].rstrip()
    img_1 = cv2.imread(frame_1, cv2.CV_8UC1)
    bbx_1 = []
    if img_1 is not None:
        bbx_1 = hd.detection_hand(proposal_pnet, refine_rnet, output_onet, img_1)
    img_2 = cv2.imread(frame_2, cv2.CV_8UC1)
    bbx_2 = []
    if img_2 is not None:
        bbx_2 = hd.detection_hand(proposal_pnet, refine_rnet, output_onet, img_2)
    with open('data/tracking_samples_%d.csv' % part_id, 'a') as file:
        if not np.asarray(bbx_1).size == 0:
            file.write('%s %f %f %f %f\n' % (cur_img, bbx_1[0], bbx_1[1], bbx_1[2], bbx_1[3]))
        if not np.asarray(bbx_2).size == 0:
            file.write('%s %f %f %f %f\n' % (cur_img, bbx_2[0], bbx_2[1], bbx_2[2], bbx_2[3]))

file.close()
sess.close()
