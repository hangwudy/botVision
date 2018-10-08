import os

image_list = []
def loadim(image_path = 'images'):
    for filename in os.listdir(image_path):
        if filename.endswith('png') and filename.find('car_door') != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list


if __name__ == '__main__':
    impath = 'images'
    IMAGE_LIST = loadim(impath)
    print(IMAGE_LIST)