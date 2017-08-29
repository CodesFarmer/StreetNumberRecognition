import numpy as np

def IoU(bbox1, bboxes):
    area_1 = (bbox1[2]-bbox1[0]+1)*(bbox1[3]-bbox1[1]+1)
    area_2 = (bboxes[:, 2]-bboxes[:, 0]+1)*(bboxes[:, 3]-bboxes[:, 1]+1)
    xx1 = np.maximum(bbox1[0], bboxes[:, 0])
    yy1 = np.maximum(bbox1[1], bboxes[:, 1])
    xx2 = np.minimum(bbox1[2], bboxes[:, 2])
    yy2 = np.minimum(bbox1[3], bboxes[:, 3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    overlap = w*h
    ovr = overlap/(area_1+area_2-overlap)
    return ovr