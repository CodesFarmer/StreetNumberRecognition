import numpy.random as npr
import numpy as np
import cv2
import os
import scipy.io as scio
import utils

data_dir = "../data"
file_path = os.path.join(data_dir, "validation_gt.mat")
#print(file_path)
mat_name = scio.loadmat(file_path)
imgnames = mat_name['imgname']
imgbboxes = mat_name['bboxes']
print(imgnames.dtype)

img_id = 5
img_path = os.path.join(data_dir, "validation", imgnames[0,img_id][0])
print(img_path)
print(imgbboxes.dtype)

bbox_index = 0
label_idnex = 4
x1_index = 0
print("label", imgbboxes[0,img_id][0][bbox_index][label_idnex][0][0])
print("x1", imgbboxes[0,img_id][0][bbox_index][x1_index][0][0])

for imname_, imgt_ in zip(imgnames[0,:], imgbboxes[0,:]):
    print(imname_[0])
    imname = imname_[0]
    imgt = imgt_[0]
    img_path = os.path.join(data_dir, "validation", imname)
    print(img_path)
    image = cv2.imread(img_path)
    height, width, channels = image.shape

    #For each number, about 20 images will be generated
    neg_num = 0

    bboxes = np.array([[]])
    for bbox_label in imgt:
        bbox_ = np.array([[bbox_label[0][0][0], bbox_label[1][0][0], bbox_label[2][0][0],bbox_label[3][0][0]]])
        bboxes = np.concatenate((bboxes, bbox_), axis=0)
    while(neg_num<20):
        size = npr.randint(12, min(height, width))
        nx = npr.randint(0, width-size)
        ny = npr.randint(0, height-size)
        crop_box = np.array([nx, ny, nx+size, ny+size])

        Iou = utils.IoU(crop_box, bboxes)
        print(bbox_label[4][0][0])
        print(bbox_label[0][0][0])