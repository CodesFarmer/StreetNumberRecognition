import xml.etree.ElementTree as xl
import numpy as np
_MAX_NUMERICAL=1e20

def read_bbx(filename):
    lx = _MAX_NUMERICAL
    ly = _MAX_NUMERICAL
    rx = -1
    ry = -1
    data = xl.parse(filename).getroot()
    for atype in data.findall('object'):
        for elem in atype.findall('polygon'):
            for point in elem.findall('pt'):
                for px in point.iter(tag='x'):
                    lx = np.min([lx, int(px.text)])
                    rx = np.max([rx, int(px.text)])
                for py in point.iter(tag='y'):
                    ly = np.min([ly, int(py.text)])
                    ry = np.max([ry, int(py.text)])

    return [lx, ly, rx, ry]