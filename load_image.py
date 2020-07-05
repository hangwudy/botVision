# coding: utf-8

# created by Hang Wu on 2018.10.07
# feedback: h.wu@tum.de

import os


def loadim(image_path = 'images', ext = 'png', key_word = 'car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list


if __name__ == '__main__':
    impath = '/home/hangwu/Repositories/AIBox/AttitudeNet/xyz/render_x_-20_20_20_y_-20_20_20_z_-90_90_45/render'
    IMAGE_LIST = loadim(impath, "png", "x")
    print(IMAGE_LIST,'\n')