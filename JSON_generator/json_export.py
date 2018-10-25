import json
from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import os

# EIGEN
from sub_mask_annotation import create_image_annotation, create_sub_mask_annotation
from sub_masks_create import create_sub_masks
from load_image import loadim

# Define which colors match which categories in the images
car_door_id = 1

category_ids = {
    1: {
        '(255, 0, 0)': car_door_id,
    },
}

is_crowd = 0

# Create the annotations
car_door_annotation = {
    'info': {
        'description': "Car Door Dataset",
        'url': "hangwudy.github.io",
        'version': '0.1',
        'year': 2018,
        'contributor': 'Hang Wu',
        'date_created': '2018/10/25',
    },
    'licenses': [
        {
        "url": "hangwudy.github.io",
        "id": 0,
        "name": 'MIT'
        }
    ],
    "images": [
        {

        }
    ],
    "annotations": [
        {

        }
    ],
    "categories": [
        {
            "supercategory": "car_parts",
            "id": 1,
            "name": 'car_door'
        }
    ]
}


def images_annotations_info(maskpath):

    annotations = []
    images = []

    mask_images_path = loadim(maskpath)
    for id_number, mask_image_path in enumerate(mask_images_path):
        file_name = mask_image_path.split(os.path.sep)[-1]
        mask_image = Image.open(mask_image_path)
        sub_masks = create_sub_masks(mask_image)
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[1][color]
            # ID number
            image_id = id_number
            annotation_id = id_number
            # image shape
            width, height = mask_image.size
            # 'images' info 
            image = create_image_annotation(file_name, width, height, image_id)
            images.append(image)
            # 'annotations' info
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
    return images, annotations

car_door_annotation['images'], car_door_annotation['annotations'] = images_annotations_info('masks')
print(json.dumps(car_door_annotation))
with open('masks/car_door.json','w') as outfile:
    json.dump(car_door_annotation, outfile)
