import numpy.random as npr
import numpy as np
import cv2
import os
import scipy.io as scio
import utils

data_dir = "../data"
negative_data_12 = data_dir + "/12/negative"
positive_data_12 = data_dir + "/12/positive"
part_data_12 = data_dir + "/12/part"
if not os.path.exists(negative_data_12):
    os.makedirs(negative_data_12, exist_ok=True)
if not os.path.exists(positive_data_12):
    os.makedirs(positive_data_12, exist_ok=True)
if not os.path.exists(part_data_12):
    os.makedirs(part_data_12, exist_ok=True)
nim_id = 0
pim_id = 0
ptim_id = 0
nfid = open(os.path.join(data_dir, "negative.txt"), 'w')
pfid = open(os.path.join(data_dir, "positive.txt"), 'w')
ptfid = open(os.path.join(data_dir, "part.txt"), 'w')
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
        size_x = npr.randint(pnet_insize, width)
        size_y = npr.randint(pnet_insize, height)
        nx = npr.randint(0, width-size_x)
        ny = npr.randint(0, height-size_y)
        crop_box = np.array([nx, ny, nx+size_x, ny+size_y])

        if(nx+size_x>width or ny+size_y>height):
            continue
        Iou = utils.IoU(crop_box, bboxes)
        crop_img = image[ny:ny+size_x, nx:nx+size_y,:]
        input_img = cv2.resize(crop_img, (pnet_insize, pnet_insize*2), interpolation=cv2.INTER_AREA)

        if(np.max(Iou)<0.3):
            #print(neg_num)
            save_path = os.path.join(negative_data_12, "%s"%nim_id+".jpg")
            nfid.write("12/negative/%s"%nim_id+" 0\n")
            cv2.imwrite(save_path, input_img)
            neg_num += 1
            nim_id += 1
    #generate the positive samples
    for box in bboxes:
        x1,y1,x2,y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        print(float(h)/float(w))

        if(w<pnet_insize or h<pnet_insize or x1<0 or y1<0): continue

        pos_num = 0
        while(pos_num<10):
            size_x = npr.randint(int(w*0.8), np.ceil(w*1.25))
            size_y = npr.randint(int(h*0.8), np.ceil(h*1.25))

            delta_x = npr.randint(-0.2*w, 0.2*w)
            delta_y = npr.randint(-0.2*h, 0.2*h)
            nx1 = int(max(x1 + w/2 + delta_x - size_x/2, 0))
            ny1 = int(max(y1 + h/2 + delta_y - size_y/2, 0))
            nx2 = int(min(x1 + size_x, width))
            ny2 = int(min(y1 + size_y, height))
            if(nx2>width or ny2>height):
                continue

            crop_box = np.array([nx1, ny1, nx2, ny2])
            crop_img = image[ny1:ny2, nx1:nx2, :]
            input_img = cv2.resize(crop_img, (pnet_insize, pnet_insize*2), interpolation=cv2.INTER_AREA)


            offset_x1 = (x1-nx1)/float(size_x)
            offset_x2 = (x2-nx2)/float(size_x)
            offset_y1 = (y1-ny1)/float(size_y)
            offset_y2 = (y2-ny2)/float(size_y)

            box_ = box.reshape(1,-1)
            Iou = utils.IoU(crop_box, box_)
            if(Iou>=0.65):
                save_path = os.path.join(positive_data_12, "%s"%pim_id+".jpg")
                pfid.write("12/positive/%s"%pim_id+" 1 %.2f %.2f %.2f %.2f\n"%(offset_x1,offset_y1,offset_x2,offset_y2))
                cv2.imwrite(save_path, input_img)
                pim_id += 1
                pos_num += 1
            elif(Iou>=0.4):
                save_path = os.path.join(part_data_12, "%s"%ptim_id+".jpg")
                ptfid.write("12/part/%s"%ptim_id+" -1 %.2f %.2f %.2f %.2f"%(offset_x1,offset_y1,offset_x2,offset_y2))
                cv2.imwrite(save_path, input_img)
                ptim_id += 1
                pos_num += 1
nfid.close()
pfid.close()
ptfid.close()