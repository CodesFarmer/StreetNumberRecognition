import numpy.random as npr
import numpy as np
import cv2
import os
import scipy.io as scio
import utils

data_dir = "../data"
negative_data_12 = data_dir + "/12/negative"
if not os.path.exists(negative_data_12):
    os.makedirs(negative_data_12, exist_ok=True)
nim_id = 0
nfid = open(os.path.join(data_dir, "negative.txt"), 'w')
pim_id = 0
file_path = os.path.join(data_dir, "validation_gt.mat")

#define the input size
pnet_insize = 12

#print(file_path)
mat_name = scio.loadmat(file_path)
imgnames = mat_name['imgname']
imgbboxes = mat_name['bboxes']

'''
Read mat examples
'''

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
    # print(imname_[0])
    imname = imname_[0]
    imgt = imgt_[0]
    img_path = os.path.join(data_dir, "validation", imname)
    # print(img_path)
    image = cv2.imread(img_path)
    height, width, channels = image.shape

    #For each number, about 20 images will be generated
    neg_num = 0

    bboxes = np.empty((0,4), int)
    for bbox_label in imgt:
        bbox_ = np.array([[np.float32(bbox_label[1][0][0]), np.float32(bbox_label[2][0][0]),
                           np.float32(bbox_label[1][0][0])+np.float32(bbox_label[3][0][0]),
                           np.float32(bbox_label[2][0][0])+np.float32(bbox_label[0][0][0])]])
        bboxes = np.append(bboxes, bbox_, axis=0)
    # cv2.rectangle(image, pt1=(np.int16(bboxes[1][0]), np.int16(bboxes[1][1])),
    #               pt2=(np.int16(bboxes[1][2]), np.int16(bboxes[1][3])), color=(255,0,0))
    # cv2.imshow("BoundingBox", image)
    # cv2.waitKey(50000)
    while(neg_num<20):
        size = npr.randint(pnet_insize, min(height, width))
        nx = npr.randint(0, width-size)
        ny = npr.randint(0, height-size)
        crop_box = np.array([nx, ny, nx+size, ny+size])

        Iou = utils.IoU(crop_box, bboxes)
        crop_img = image[ny:ny+size, nx:nx+size,:]
        input_img = cv2.resize(crop_img, (pnet_insize, pnet_insize), interpolation=cv2.INTER_LINEAR)

        if(np.max(Iou)<0.3):
            print(neg_num)
            save_path = os.path.join(negative_data_12, "%s"%nim_id+".jpg")
            nfid.write("12/negative/%s"%nim_id+" 0\n")
            cv2.imwrite(save_path, input_img)
            neg_num += 1
            nim_id += 1
        # print(bbox_label[4][0][0])
        # print(bbox_label[0][0][0])