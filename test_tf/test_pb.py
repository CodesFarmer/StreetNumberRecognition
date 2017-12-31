import tensorflow as tf
import cv2
import glob
import time
import numpy as np
import neuralnetwork.detect_geometry as dg

frozen_graph = "../model/pnet.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
        )
# output = graph.get_tensor_by_name("pnet/coor/Add:0")
# output.append(graph.get_tensor_by_name("pnet/prob_sm:0"))
sess = tf.Session(graph=graph)
proposal_pnet = lambda x: sess.run(("pnet/coor/Add:0", "pnet/prob_sm:0"), feed_dict={"pnet/Placeholder:0":x})
minsize = 12
factor = 0.709 # scale factor
threshold = [0.7, 0.6, 0.6]
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


    img_show = img_ori.copy()
    for k in range(0, numbox):
        cv2.rectangle(img_show, (int(total_boxes[k, 0]), int(total_boxes[k, 1])), (int(total_boxes[k, 2]), int(total_boxes[k, 3])), (255, 255, 255), 1)
    cv2.imshow("bbx by pnet", img_show)
    cv2.waitKey(0)