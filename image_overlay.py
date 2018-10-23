# coding: utf-8

# created by Hang Wu on 2018.10.07
# feedback: h.wu@tum.de


import cv2
import numpy as np
from numpy import random
import os
# Eigen
import load_image
import generate_dict


def overlap(background, foreground, bnd_pos):
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    # print(foreground.shape)
    # print(background.shape)

    rows, cols = foreground.shape[:2]
    rows_b, cols_b = background.shape[:2]

    # Mask initialization
    object_mask = np.zeros([rows_b, cols_b, 3], np.uint8)

    # Range of x and y
    low_x = bnd_pos['xmin']
    low_y = bnd_pos['ymin']
    high_x = bnd_pos['xmax']
    high_y = bnd_pos['ymax']
    # Movement for random position
    move_x = int(random.randint(- low_x, cols_b - high_x, 1))
    move_y = int(random.randint(- low_y, rows_b - high_y, 1))
    # move_y = random.randint(rows_b - high_y -1, rows_b - high_y, 1)

    print('movement x:',move_x)
    # print(high_y)
    print('movement y:',move_y)
    
    for i in range(rows):

        ###### for solid mask 1. part >>>>
        col_min = cols
        col_max = 0
        ###### for solid mask 1. part <<<<

        for j in range(cols):
            if foreground[i,j][3] != 0:
                # Overlap images
                try:
                    background[i + move_y, j + move_x] = foreground[i,j]
                except:
                    break
                # Mask generating (for nomal mask with hols)
                # object_mask[i + move_y, j + move_x] = [0, 0, 255]

                ###### for solid mask 2. part >>>>
                if col_min > j:
                    col_min = j
                if col_max < j:
                    col_max = j
        for col in range(col_min, col_max + 1):
            try:
                object_mask[i + move_y, col + move_x] = [0, 0, 255]
            except:
                break
                ###### for solid mask 2. part <<<<

    output_image = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)


    save_name = bnd_pos['filename'][:-4]
    # Path
    current_path = os.path.abspath('.')
    save_path = os.path.join(os.path.abspath(os.path.dirname(current_path) + 
                    os.path.sep + "data/images"), '{}.jpg'.format(save_name))
    print(save_path)
    # Update xml data
    ## file info
    bnd_pos['folder'] = save_path.split(os.path.sep)[-2]
    bnd_pos['filename'] = save_path.split(os.path.sep)[-1]
    bnd_pos['path'] = save_path
    ## image info
    rows_out, cols_out, channels_out = output_image.shape
    bnd_pos['width'] = cols_out
    bnd_pos['height'] = rows_out
    bnd_pos['depth'] = channels_out
    ## x-y value
    bnd_pos['xmin'] += move_x
    bnd_pos['ymin'] += move_y
    bnd_pos['xmax'] += move_x
    bnd_pos['ymax'] += move_y
    # test
    print(bnd_pos)


    # Save images
    cv2.imwrite('../data/images/{}.jpg'.format(save_name), output_image)
    cv2.imwrite('../data/annotations/masks/{}.png'.format(save_name), object_mask)

    # Display
    # cv2.imshow('{}.jpg'.format(save_name), output_image)
    # cv2.imshow('mask', object_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return bnd_pos


if __name__ == '__main__':
    
    fg_list = load_image.loadim('images')
    print(fg_list)
    bg_list = load_image.loadim('background','jpg','Fabrik')
    print(bg_list)
    for fg in fg_list:
        bnd_info = generate_dict.object_dict(fg)
        fg = cv2.imread(fg, -1)
        bg_path = random.choice(bg_list)
        print(bg_path)
        bg = cv2.imread(bg_path, -1)
        overlap(bg, fg, bnd_info)
