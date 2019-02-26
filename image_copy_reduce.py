import shutil
import os
import re

# get image absolut path
def loadim(image_path = 'Car_Door', ext = 'png', key_word = 'car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list


original_path = "/media/hangwu/TOSHIBA_EXT/Dataset/renderings_square"

car_door_list = loadim(image_path=original_path)

for car_door_img in car_door_list:
        img_name = car_door_img.split(os.path.sep)[-1]
        match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', img_name, re.I)
        latitude = int(match.groups()[2])
        longitude = int(match.groups()[4])
        if latitude % 2 == 0 and longitude % 2 == 0:
                path_origin = car_door_img
                path_destination = '/media/hangwu/TOSHIBA_EXT/Dataset/renderings_square_half'
                shutil.copy2(path_origin, path_destination)



