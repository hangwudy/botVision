import cv2
import os
import json
import numpy as np

def loadim(image_path, ext = 'png'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext):
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list



def object_dict(impath):
    # initialize the dictionary
    bnd_dict = {'imageName':None, 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    # read the image
    img = cv2.imread(impath, -1)
    rows, cols = img.shape[:2]
    name = impath.split(os.path.sep)[-1]
    print(name)
    bnd_dict['imageName'] = name

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
            # if px[0] == 28 and px[1] == 130 and px[2] == 49:
            #     print("Green")

    bnd_dict['xmin'] = xmin
    bnd_dict['ymin'] = ymin
    bnd_dict['xmax'] = xmax
    bnd_dict['ymax'] = ymax
    # IMPORTANT: .COPY()
    # bnd_position.append(bnd_dict.copy())
    # tmp = img[xmin:xmax, ymin:ymax]
    # cv2.imshow("", tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return bnd_dict, img


def crop_to_save(d, save_path, image_path):
    # Bounding Box information >>>
    save_name = d.get('imageName')
    xmin = d.get('xmin')
    ymin = d.get('ymin')
    xmax = d.get('xmax')
    ymax = d.get('ymax')
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    side_len_diff_half = int(abs(bbox_height - bbox_width) / 2)

    image = cv2.imread(image_path)

    crop_image = image[ymin:ymax, xmin:xmax]


    if bbox_height >= bbox_width:
        new_patch = np.zeros((bbox_height, bbox_height ,3), np.uint8)
        for row in range(crop_image.shape[0]):
            for col in range(crop_image.shape[1]):
                px = crop_image[row, col]
                if px[0] == 28 and px[1] == 130 and px[2] == 49:
                    new_patch[row, col + side_len_diff_half] = [0, 0, 0]
                else:
                    new_patch[row, col + side_len_diff_half] = px
    else:
        new_patch = np.zeros((bbox_width, bbox_width ,3), np.uint8)
        for row in range(crop_image.shape[0]):
            for col in range(crop_image.shape[1]):
                px = crop_image[row, col]
                if px[0] == 28 and px[1] == 130 and px[2] == 49:
                    new_patch[row + side_len_diff_half, col] = [0, 0, 0]
                else:
                    new_patch[row + side_len_diff_half, col] = px
    
    cv2.imwrite("{}/{}".format(save_path, save_name), new_patch)

def all_in_one():
    # initialize the dictionary
    bnd_dict = {'imageName':None, 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    # read the image
    img = cv2.imread(impath, -1)
    rows, cols = img.shape[:2]
    name = impath.split(os.path.sep)[-1]
    print(name)
    bnd_dict['imageName'] = name

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
            if px[1] == 255:
                print("Green")

    bnd_dict['xmin'] = xmin
    bnd_dict['ymin'] = ymin
    bnd_dict['xmax'] = xmax
    bnd_dict['ymax'] = ymax
    # IMPORTANT: .COPY()
    # bnd_position.append(bnd_dict.copy())
    # tmp = img[xmin:xmax, ymin:ymax]
    # cv2.imshow("", tmp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    save_name = d.get('imageName')
    xmin = d.get('xmin')
    ymin = d.get('ymin')
    xmax = d.get('xmax')
    ymax = d.get('ymax')
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    side_len_diff_half = int(abs(bbox_height - bbox_width) / 2)

    image = cv2.imread(image_path)

    crop_image = image[ymin:ymax, xmin:xmax]


    if bbox_height >= bbox_width:
        new_patch = np.zeros((bbox_height, bbox_height ,3), np.uint8)
        for row in range(crop_image.shape[0]):
            for col in range(crop_image.shape[1]):
                new_patch[row, col + side_len_diff_half] = crop_image[row, col]
    else:
        new_patch = np.zeros((bbox_width, bbox_width ,3), np.uint8)
        for row in range(crop_image.shape[0]):
            for col in range(crop_image.shape[1]):
                new_patch[row + side_len_diff_half, col] = crop_image[row, col]
    
    cv2.imwrite("{}/{}".format(save_path, save_name), new_patch)


if __name__ == "__main__":
    imlist = loadim('xyz/render/render')
    res = []
    _ = 0
    n = len(imlist)
    for impath in imlist:
        d, im = object_dict(impath)
        crop_to_save(d, 'xyz/render/cropped', impath)
        res.append(d)
        if _ % 100 == 0:
            print('{:.2f}% finished'.format(_/n*100))
        _ += 1
    with open("res.json", "w") as f:
        json.dump(res, f)
