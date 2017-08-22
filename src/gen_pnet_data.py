import numpy as np
from scipy import io
import h5py
import os
import scipy.io as scio

data_dir = "../data"
file_path = os.path.join(data_dir, "validation.mat")
#print(file_path)
mat_name = scio.loadmat(file_path)
imgnames = mat_name['imgname']
print(imgnames.dtype)
img_path = os.path.join(data_dir, "train", imgnames[0,1][0])
print(img_path)
imgbboxes = mat_name['bboxes']
print(imgbboxes.dtype)

bbox_index = 0;
label_idnex = 4;
x1_index = 0;
print("label", imgbboxes[0,1][0][bbox_index][label_idnex][0][0])
print("x1", imgbboxes[0,1][0][bbox_index][x1_index][0][0])


#file_path = os.path.join(data_dir, "train_gt.mat")
#mat_gt = h5py.File(file_path)
