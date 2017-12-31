import xml.etree.ElementTree as xl
import glob

# data = xl.parse('../data/1509071298176172252.xml').getroot()
# for atype in data.findall('object'):
#     for elem in atype.findall('polygon'):
#         for point in elem.findall('pt'):
#             for px in point.iter(tag='x'):
#                 print('x', px.text)
#             for py in point.iter(tag='y'):
#                 print('y', py.text)
#             # print(point.iter(tag='x').text)
#             # for xy in point.findall('x'):
#             #     print(xy.get(''))
#             # print(point('x'))

# print(glob.glob('/home/slam/datasets/handdetect_sample/01/1/cam0/*.png'))
with open('/home/slam/datasets/handdetect_sample/01/ir_depth.txt', 'r') as file:
    content = file.readlines()

for line in content:
    print(line)