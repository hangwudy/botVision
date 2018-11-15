import json
import os
import re


from load_image import loadim


pose_annotations = {
    "annotations": [
        {

        }
    ]
}

def images_pose_anno(image_path):
    file_name = image_path.split(os.path.sep)[-1]
    pose_la, pose_lo = get_pose_from_filename(file_name)
    annotation = {
        "image_id": file_name,
        "latitude": pose_la,
        "longitude": pose_lo
    }
    return annotation


def get_pose_from_filename(file_name):
   # car_door_1_125.png ==>> 'car_door', 1, 125
    match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', file_name, re.I)
    class_name = match.groups()[0]
    latitude = match.groups()[2]
    longitude = match.groups()[4]
    # print(class_name, latitude, longitude)
    return latitude, longitude    

def anno_export(img_dir):
    image_path_list = loadim(img_dir)
    annotations = []
    for image_path in image_path_list:
        annotation = images_pose_anno(image_path)
        annotations.append(annotation)
    pose_annotations['annotations'] = annotations
    print(pose_annotations)
    return pose_annotations

def read_pose_json(json_path):
    anno_file = open(json_path, 'r')
    anno_list = json.load(anno_file)

    for d in anno_list['annotations']:
        print(d)
        f = d.get('longitude')
        print(f)

def main():
    car_door_pose = anno_export('masks')
    with open('car_door_pose.json', 'w') as outfile:
        json.dump(car_door_pose, outfile)


if __name__ == '__main__':
    # main()
    read_pose_json('car_door_pose.json')


