# coding: utf-8

# created by Hang Wu on 2018.10.07
# feedback: h.wu@tum.de


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Eigen
import load_image

def object_dict(impath, scale = 0.4):
    # initialize the dictionary
    bnd_dict = {'folder': 'FOLDER','filename':'NAME', 'path': 'PATH', 'width': 0, 'height': 0, 'depth': 0,
                'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    # read the image
    img = cv2.imread(impath, -1)
    img = cv2.resize(img, (0,0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)

    rows, cols, channels = img.shape

    # Initialization for the position
    xmin = cols
    ymin = rows
    xmax = 0
    ymax = 0

    for i in range(rows):
        for j in range(cols):
            px = img[i,j]
            if px[3] !=0:
                if ymin > i:
                    ymin = i
                if xmin > j:
                    xmin = j
                if ymax < i:
                    ymax = i
                if xmax < j:
                    xmax = j

    # Bounding box information for .xml
    bnd_dict['folder'] = impath.split(os.path.sep)[-2]
    bnd_dict['filename'] = impath.split(os.path.sep)[-1]
    bnd_dict['path'] = impath
    bnd_dict['width'] = cols
    bnd_dict['height'] = rows
    bnd_dict['depth'] = channels
    bnd_dict['xmin'] = xmin
    bnd_dict['ymin'] = ymin
    bnd_dict['xmax'] = xmax
    bnd_dict['ymax'] = ymax
    # IMPORTANT: .COPY()
    # bnd_position.append(bnd_dict.copy())
    return bnd_dict

if __name__ == '__main__':
    # test
    image_path_list = load_image.loadim('../data/car_door')
    for image_path in image_path_list:
        bp = object_dict(image_path)
        print(bp)