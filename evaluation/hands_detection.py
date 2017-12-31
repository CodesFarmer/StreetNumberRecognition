import tensorflow as tf
from tensorflow.python.client import timeline
import cv2
import numpy as np
import os
import time
import glob

import neuralnetwork.handetector as hd
import neuralnetwork.detect_geometry as dg

root_dir = '/home/slam/TestRoom/tensorflow_ep/handetect'

minsize = 12
factor = 0.709 # scale factor
model_path = []
# model_path.append('model/model_pnet.ckpt')
# model_path.append('model/model_rnet.ckpt')
# model_path.append('model/model_onet.ckpt')
model_path.append('%s/model/pnet.pb'%root_dir)
model_path.append('%s/model/rnet.pb'%root_dir)
# model_path.append('model/onet.pb')
# model_path.append('model/onet_quantized.pb')
# model_path.append('model/onet_mn_pool.pb')
# model_path.append('model/onet_mn_distilling_01_1_1_0_05_134.pb')
# model_path.append('model/onet_mn_distilling_pool.pb')
# model_path.append('model/onet_mn_sc_pool.pb')
model_path.append('%s/model/onet_mn_wl75_rw3.pb'%root_dir)
whichnet = 'onet'
whichop = 'mnwlr1'
# model_path.append('model/model_onetmn.ckpt')
threshold = [0.7, 0.6, 0.6]

sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
params = {'options': run_options, 'metadata': run_metadata}
# proposal_pnet, refine_rnet, output_onet = hd.create_dnn(sess, model_path, params)
proposal_pnet, refine_rnet, output_onet = hd.load_dnn(sess, model_path, params)
# proposal_pnet, refine_rnet, output_onet = hd.create_dn(sess, model_path)
# proposal_pnet = hd.create_pnet(sess, '%s/model/model/model_pnet.ckpt' % root_dir, '%s/graph/model_pnet.pb' % root_dir)
# proposal_pnet = hd.create_rnet(sess, 'model/model/model_rnet.ckpt')
# proposal_pnet = hd.create_onet(sess, 'model/model/model_onet.ckpt')
# rnet = hd.create_rnet(sess, model_path[1])
# onet = hd.create_onet(sess, 'model/model/model_onet.ckpt')

imglists = glob.glob('../data/*.png')

for img_name in imglists:
    img_ori = cv2.imread(img_name, cv2.CV_8UC1)
    img = img_ori.astype(float)
    # # img = np.expand_dims(img, -1)
    # img = img*0.0125
    # img = [[img]]
    # img = np.transpose(img, axes=[0,2,3,1])

    # output = proposal_pnet(img)
    # print(output)
    time_before = time.time()

    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = np.empty(0)
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # creat scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for j in range(len(scales)):
        scale = scales[j]
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = dg.imresample(img, (hs, ws))
        im_data = (im_data - 0.0) * 0.0125
        im_data = [[im_data]]
        img_y = np.transpose(im_data, axes=[0,2,3,1])
        out = proposal_pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = dg.generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])
        boxes[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]] = boxes[:, [1, 0, 3, 2, 4, 6, 5, 8, 7]]

        # inter-scale nms
        pick = dg.nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = dg.nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = dg.rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = dg.pad(total_boxes.copy(), w, h)


    # img_show = img_ori.copy()
    # for k in range(0, numbox):
    #     cv2.rectangle(img_show, (int(total_boxes[k, 0]), int(total_boxes[k, 1])), (int(total_boxes[k, 2]), int(total_boxes[k, 3])), (255, 255, 255), 1)
    # cv2.imshow("bbx by pnet", img_show)
    # cv2.waitKey(0)


    numbox = total_boxes.shape[0]
    print("RNET: There are %d bounding boxes" % numbox)
    if numbox>0:
        # second stage
        tempimg = np.zeros((24,24,numbox))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k])))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k]] = img[y[k]-1:ey[k],x[k]-1:ex[k]]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,k] = dg.imresample(tmp, (24, 24))
            else:
                raise RuntimeError("Empty input for rnet")
        tempimg = (tempimg-0)*0.0125
        tempimg = [tempimg]
        tempimg1 = np.transpose(tempimg, (3,1,2,0))
        out = refine_rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]
        ipass = np.where(score>threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]
        mv[[0, 1, 2, 3], :] = mv[[1, 0, 3, 2], :]
        if total_boxes.shape[0]>0:
            pick = dg.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = dg.bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = dg.rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    # img_show = img_ori.copy()
    # for k in range(0, numbox):
    #     cv2.rectangle(img_show, (int(total_boxes[k, 0]), int(total_boxes[k, 1])), (int(total_boxes[k, 2]), int(total_boxes[k, 3])), (255, 255, 255), 1)
    # cv2.imshow("bbx by rnet", img_show)
    # cv2.waitKey(0)



    numbox = total_boxes.shape[0]
    print("ONET: There are %d bounding boxes" % numbox)
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = dg.pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k])))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k]] = img[y[k] - 1:ey[k], x[k] - 1:ex[k]]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, k] = dg.imresample(tmp, (48, 48))
            else:
                raise RuntimeError("Empty input for onet")
        tempimg = (tempimg-0)*0.0125
        tempimg = [tempimg]
        tempimg1 = np.transpose(tempimg, (3,1,2,0))
        out = output_onet(tempimg1)
        out0 = np.transpose(out[0])
        out2 = np.transpose(out[1])
        score = out2[1, :]
        # points = out1
        ipass = np.where(score > threshold[2])
        # points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]
        mv[[0, 1, 2, 3], :] = mv[[1, 0, 3, 2], :]
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        # points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        # points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = dg.bbreg(total_boxes.copy(), np.transpose(mv))
            pick = dg.nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            # points = points[:, pick]
    numbox = total_boxes.shape[0]
    time_after = time.time()
    print("time consuming is %f" % (time_after - time_before))

    tl = timeline.Timeline(run_metadata.step_stats)
    # # print(tl)
    # ctf = tl.generate_chrome_trace_format()
    # if not os.path.exists('time_consuming/data'):
    #     os.makedirs('time_consuming/data', exist_ok=True)
    # prefix = '%s_%s_%s' % (img_name, whichop, whichnet)
    # with open('time_consuming/%s.json' % prefix, 'w') as file:
    #     file.write(ctf)


    img_show = img_ori.copy()
    for k in range(0, numbox):
        cv2.rectangle(img_show, (int(total_boxes[k, 0]), int(total_boxes[k, 1])), (int(total_boxes[k, 2]), int(total_boxes[k, 3])), (255, 255, 255), 1)
    cv2.imshow("bbx by onet", img_show)
    cv2.waitKey(0)
    # if not os.path.exists('time_consuming/image/data'):
    #     os.makedirs('time_consuming/image/data', exist_ok=True)
    # cv2.imwrite('time_consuming/image/%s.png' % prefix, img_show)