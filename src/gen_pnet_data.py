import numpy as np
from scipy import io
import h5py
import os
import scipy.io as scio

data_dir = "/home/lowell/MachineLearning/Dataset/StreetNumber"
file_path = os.path.join(data_dir, "train", "digitStruct.mat")
print(file_path)
mat_data = h5py.File(file_path)
print(mat_data.keys())
print(mat_data.values())
for k,v in mat_data.items():
    print(k)