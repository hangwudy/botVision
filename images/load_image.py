# coding: utf-8

# created by Hang Wu on 2018.10.07
# feedback: h.wu@tum.de

import os
import cv2
import numpy as np

def loadim(image_path = 'images', ext = 'png', key_word = 'car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list

def mask_gen(img_path):
    
    img = cv2.imread(img_path, -1)
    rows, cols = img.shape[:2]
    object_mask = np.zeros([rows, cols, 3], np.uint8)
    for row in range(rows):
        col_min = cols
        col_max = 0
        for col in range(cols):
            # col_min = cols
            # col_max = 0
            if img[row, col][3]!=0:
                if col_min > col:
                    col_min = col
                if col_max < col:
                    col_max = col
        # print(col_min, col_max)
        for j in range(col_min, col_max + 1):
            object_mask[row, j] = [0, 0, 255]
    save_name = 'test_mask'
    cv2.imwrite('{}.png'.format(save_name), object_mask)

def mask_generate(img_RGBA,rows_b,cols_b, move_x, move_y):
    
    img = cv2.imread(img_RGBA, -1)
    rows, cols = img.shape[:2]
    object_mask = np.zeros([rows_b, cols_b, 3], np.uint8)
    for row in range(rows):
        col_min = cols
        col_max = 0
        for col in range(cols):
            # col_min = cols
            # col_max = 0
            if img[row, col][3]!=0:
                if col_min > col:
                    col_min = col
                if col_max < col:
                    col_max = col
        # print(col_min, col_max)
        for j in range(col_min, col_max + 1):
            object_mask[row + move_y, j + move_x] = [0, 0, 255]
    save_name = 'test_mask'
    cv2.imwrite('{}.png'.format(save_name), object_mask)



if __name__ == '__main__':
    impath = '.'
    IMAGE_LIST = loadim(impath, 'png', '1')
    print(IMAGE_LIST,'\n')
    mask_gen(IMAGE_LIST[0])