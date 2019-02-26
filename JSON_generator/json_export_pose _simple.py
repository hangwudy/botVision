import json
import os
import re
import cv2

from load_image import loadim


pose_annotations = {
    "annotations": [
        {

        }
    ]
}


def object_dict(impath):
    # initialize the dictionary
    bnd_dict = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
    # read the image
    img = cv2.imread(impath, -1)
    rows, cols = img.shape[:2]

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

    bnd_dict['xmin'] = xmin
    bnd_dict['ymin'] = ymin
    bnd_dict['xmax'] = xmax
    bnd_dict['ymax'] = ymax
    # IMPORTANT: .COPY()
    # bnd_position.append(bnd_dict.copy())
    return bnd_dict


def images_pose_anno(image_path):
    file_name = image_path.split(os.path.sep)[-1]
    pose_la, pose_lo = get_pose_from_filename(file_name)
    # bbox_dict = object_dict(image_path)
    # xmin, ymin, xmax, ymax = bbox_dict["xmin"], bbox_dict["ymin"], bbox_dict["xmax"], bbox_dict["ymax"]
    annotation = {
        "image_id": file_name,
        "latitude": pose_la,
        "longitude": pose_lo,
        # "xmin": xmin,
        # "ymin": ymin,
        # "xmax": xmax,
        # "ymax": ymax
    }
    return annotation


def get_pose_from_filename(file_name):
    # car_door_1_125.png ==>> 'car_door', 1, 125
    match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', file_name, re.I)
    # class_name = match.groups()[0]
    latitude = int(match.groups()[2])
    longitude = int(match.groups()[4])
    # print(class_name, latitude, longitude)
    return latitude, longitude    


def anno_export(img_dir):
    image_path_list = loadim(img_dir)
    annotations = []
    i = 1
    print("begin to load.")
    for image_path in image_path_list:
        annotation = images_pose_anno(image_path)
        annotations.append(annotation)
        if i%100 == 0:
            print("{:.2f}% finished.".format(i/len(image_path_list)*100))
            print(annotation)
        i += 1
    print("all images loaded.")
    pose_annotations['annotations'] = annotations
    # print(pose_annotations)
    return pose_annotations


def read_pose_json(json_path):
    anno_file = open(json_path, 'r')
    anno_list = json.load(anno_file)

    for d in anno_list['annotations']:
        print(d)
        f = d.get('longitude')
        print(f)


def main(dataset_path, save_dir):
    car_door_pose = anno_export(dataset_path)
    print("start to write file.")
    with open('{}/car_door_attitude_half.json'.format(save_dir), 'w') as outfile:
        json.dump(car_door_pose, outfile)
    print("writing finished!")


if __name__ == '__main__':

    main('/media/hangwu/TOSHIBA_EXT/Dataset/renderings_square_half', '/media/hangwu/TOSHIBA_EXT/Dataset/annotations/pose')
    # read_pose_json('car_door_pose.json')


