# JSON codebook generator
# Attitude: Diagonal Length
# @author: Hang Wu
# @date: 2019.02.11

import json
import numpy as np

"""
Codebook Format:

{
    "<latitude_1>": {
        "<longitude_1>": <diagonal_length_1>,
        "<longitude_2>": <diagonal_length_2>,
        ...
        "<longitude_360>": <diagonal_length_360>,
    },
    "<latitude_2>": {
        "<longitude_1>": <diagonal_length_1>,
        "<longitude_2>": <diagonal_length_2>,
        ...
        "<longitude_360>": <diagonal_length_360>,
    },

    ...

    "<latitude_45>": {
        "<longitude_1>": <diagonal_length_1>,
        "<longitude_2>": <diagonal_length_2>,
        ...
        "<longitude_360>": <diagonal_length_360>,
    }
}

"""


def data_extraction(annotation):

    latitude = annotation["latitude"]
    longitude = annotation["longitude"]
    xmin = annotation["xmin"]
    ymin = annotation["ymin"]
    xmax = annotation["xmax"]
    ymax = annotation["ymax"]

    bbox_width = abs(xmax - xmin)
    bbox_height = abs(ymax - ymin)

    bbox_diagonal_length = np.sqrt(np.power(bbox_width, 2) + np.power(bbox_height, 2))
    # limit floats to two decimal points
    bbox_diagonal_length = round(bbox_diagonal_length, 2)

    return latitude, longitude, bbox_diagonal_length


def dict_generation(json_path="car_door_pose.json"):

    with open(json_path) as jf:
        json_data = json.load(jf)
    # dict annotation first layer
    dict_annotation_1 = {}
    for annotation in json_data["annotations"]:
        # latitude, longitude, diagonal length
        dict_1, dict_2, dict_3 = data_extraction(annotation)
        if not dict_1 in dict_annotation_1:
            # dict annotation second layer
            dict_annotation_2 = {}
        else:
            # dict annotation second layer
            dict_annotation_2 = dict_annotation_1[dict_1]
        dict_annotation_2.setdefault(dict_2, dict_3)
        dict_annotation_1.setdefault(dict_1, dict_annotation_2)

    return dict_annotation_1

def write_dict_to_file(json_path="car_door_pose.json", save_name="car_door_codebook.json"):
    dict_output = dict_generation(json_path)
    with open(save_name, "w") as json_output:
        json.dump(dict_output, json_output)
    print("Data saved in {}".format(save_name))

def read_dict_file(json_path="car_door_codebook.json"):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    
    print(json_data)
    return json_data


if __name__ == "__main__":
    dict_annotation = dict_generation()
    print(dict_annotation)
    write_dict_to_file()
    test_dict = read_dict_file()
    print(test_dict["2"]["210"])

    

