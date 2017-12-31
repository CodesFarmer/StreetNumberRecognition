import tensorflow as tf
from tensorflow.python.client import timeline
import cv2
import numpy as np
import os
import time
import neuralnetwork.read_text as rt

import neuralnetwork.handetector as hd
import neuralnetwork.detect_geometry as dg

minsize = 12
factor = 0.709 # scale factor
model_path = []
model_path.append('../model/pnet.pb')
model_path.append('../model/rnet.pb')
# model_path.append('model/onet.pb')
# model_path.append('model/onet_quantized.pb')
# model_path.append('model/onet_mn_pah_pool.pb')
# model_path.append('model/onet_mn_distilling_01_1_1_0_05_134.pb')
# model_path.append('model/onet_mn_distilling_pool.pb')
# model_path.append('model/onet_mn_sc_pool.pb')
model_path.append('../model/onet_mn_wl75_rw3.pb')
whichnet = 'onet'
whichop = 'r3tracking'
# model_path.append('model/model_onetmn.ckpt')
threshold = [0.7, 0.6, 0.6]

sess = tf.Session()
# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# run_metadata = tf.RunMetadata()
# params = {'options': run_options, 'metadata': run_metadata}
# proposal_pnet, refine_rnet, output_onet = hd.create_dnn(sess, model_path, params)
params = []
proposal_pnet, refine_rnet, output_onet = hd.load_dnn(sess, model_path, params)

#load the file lists
with open('/home/slam/datasets/handdetect_sample/01/image_stream.txt', 'r') as file:
    content = file.readlines()
total_num = len(content)

iter = 0
prob = 0.0
tracking_threshold = 0.9
temp_box = []
time_new_round = time.time()

time_start = time.time()
for img_name in content:
    img_name = img_name.rstrip()
    img_ori = cv2.imread(img_name, cv2.CV_8UC1)
    img = img_ori.astype(float)
    time_end = time.time()
    while time_end - time_start < 0.05:
        time_end = time.time()
    time_start = time_end

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

    time_before_total = time.time()
    if prob < tracking_threshold:
        time_consuming_pnet = 0
        # first stage
        for j in range(len(scales)):
            scale = scales[j]
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = dg.imresample(img, (hs, ws))
            im_data = (im_data - 0.0) * 0.0125
            im_data = [[im_data]]
            img_y = np.transpose(im_data, axes=[0,2,3,1])
            time_before = time.time()
            out = proposal_pnet(img_y)
            time_after = time.time()
            time_consuming_pnet = (time_after - time_before) + time_consuming_pnet
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

        numbox = total_boxes.shape[0]
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
            time_before = time.time()
            out = refine_rnet(tempimg1)
            time_after = time.time()
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
        time_consuming_rnet = (time_after - time_before)

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
            time_before = time.time()
            out = output_onet(tempimg1)
            time_after = time.time()
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
        time_consuming_onet = (time_after - time_before)
    #tracking
    else:
        # third stage
        time_consuming_pnet = 0
        time_consuming_rnet = 0
        numbox = 1
        total_boxes = np.fix([temp_box]).astype(np.int32)
        total_boxes = dg.rerec(total_boxes.copy())
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
        time_before = time.time()
        out = output_onet(tempimg1)
        time_after = time.time()
        out0 = np.transpose(out[0])
        out2 = np.transpose(out[1])
        score = out2[1, :]
        # # points = out1
        # ipass = np.where(score > threshold[2])
        # # points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[:, 0:4].copy(), np.expand_dims(score, 1)])
        mv = out0[:, :]
        mv[[0, 1, 2, 3], :] = mv[[1, 0, 3, 2], :]
        total_boxes = dg.bbreg(total_boxes.copy(), np.transpose(mv))

    numbox = total_boxes.shape[0]
    # if
    time_consuming_onet = (time_after - time_before)


    time_after_total = time.time()
    time_consuming_total = time_after_total - time_before_total
    iter = iter + 1
    if iter % 100 == 0:
        print("%d/%d" % (iter, total_num))
    prob = 0.0
    if numbox > 0:
        total_boxes = total_boxes[0, :]
        xml_name = img_name
        xml_name = xml_name.replace('png', 'xml')
        xml_name = xml_name.replace('cam0', 'xml')
        if os.path.exists(xml_name):
            bbx = rt.read_bbx(xml_name)
            area_iou = dg.IoU(bbx, total_boxes)
            with open('../time_consuming/%s_quality.txt' % whichop, 'a') as file:
                file.write('%f %f %f %f %f\n' % (area_iou, time_consuming_pnet, time_consuming_rnet, time_consuming_onet, time_consuming_total))


        # img_show = img_ori.copy()
        img_show = cv2.cvtColor(img_ori, cv2.COLOR_GRAY2RGB)
        # print(dg.IoU(bbx, total_boxes[0, :]))
        # cv2.rectangle(img_show, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), (0, 255, 0), 1)
        cv2.rectangle(img_show, (int(total_boxes[0]), int(total_boxes[1])),
                      (int(total_boxes[2]), int(total_boxes[3])), (0, 0, 255), 1)

        if len(temp_box) ==0:
            temp_box = total_boxes
        cv2.rectangle(img_show, (int(temp_box[0]), int(temp_box[1])),
                      (int(temp_box[2]), int(temp_box[3])), (0, 255, 0), 1)

        # xx1 = int(np.maximum(bbx[0], total_boxes[0, 0]))
        # yy1 = int(np.maximum(bbx[1], total_boxes[0, 1]))
        # xx2 = int(np.minimum(bbx[2], total_boxes[0, 2]))
        # yy2 = int(np.minimum(bbx[3], total_boxes[0, 3]))
        # cv2.rectangle(img_show, (xx1, yy1), (xx2, yy2), (0, 255, 0), 1)

        # if area_iou < 0.7:
        #     _, filename = os.path.split(img_name)
        #     file_prefix, _ = os.path.splitext(filename)
        #     dest_path = os.path.join('data/failure_case', '%s.png' % file_prefix)
        #     if not os.path.exists('data/failure_case'):
        #         os.makedirs('data/failure_case', exist_ok=True)
        #     # print(dest_path)
        #     cv2.imwrite(dest_path, img_show)

        cv2.imshow("bbx by onet", img_show)
        cv2.waitKey(1)


        prob = total_boxes[4]
        temp_box = total_boxes
    else:
        with open('../time_consuming/%s_failure.txt' % whichop, 'a') as file:
            file.write('%s\n' % img_name)

    time_fragment = time.time() - time_new_round
    if time_fragment > 1.0:
        prob = 1.0
        time_new_round = time.time()