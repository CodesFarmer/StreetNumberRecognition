import cv2
import neuralnetwork.detect_geometry as dg
import os

#
# path = 'path/to/where/idont/know/either.file'
#
# print(os.path.split(path))

for i in range(0, 10):
    print(i)

img_name = '../data/1508741398926199885.png'
im = cv2.imread(img_name, cv2.CV_8UC1)
cv2.imshow("ORIGINAL", im)
cv2.waitKey(0)
dg.candidate_region(im)

if os.path.exists('../data/1508741398926199885.png'):
    print('File is existing!')
else:
    print('File does not existing!')