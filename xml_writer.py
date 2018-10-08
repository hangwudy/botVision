# coding:utf-8

from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import cv2

# Eigen
import image_overlay
import load_image
import generate_dict

def xml_generator(bndbox):
    # Root
    node_root = Element('annotation')
    ## Folder
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = bndbox['folder']
    ## Filename
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = bndbox['filename']
    ## Path
    node_path = SubElement(node_root, 'path')
    node_path.text = bndbox['path']
    ## Source
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'
    ## Size
    node_size = SubElement(node_root, 'size')
    ### Width
    node_width = SubElement(node_size, 'width')
    node_width.text = str(bndbox['width'])
    ### Height
    node_height = SubElement(node_size, 'height')
    node_height.text = str(bndbox['height'])
    ### Depth
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(bndbox['depth'])
    ## Segmented
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    ## Object
    node_object = SubElement(node_root, 'object')
    ### Name
    node_name = SubElement(node_object, 'name')
    node_name.text = 'car_door'
    ### Pose
    node_pose = SubElement(node_object, 'pose')
    node_pose.text = 'Unspecified'
    ### Difficult
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    ### Bounding box
    node_bndbox = SubElement(node_object, 'bndbox')
    #### x-y value
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(bndbox['xmin'])
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(bndbox['ymin'])
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(bndbox['xmax'])
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(bndbox['ymax'])
    # format display
    xml = tostring(node_root, pretty_print=True)
    xml_name = bndbox['filename'][:-4]
    fp = open('xmls/{}.xml'.format(xml_name), 'w')
    fp.write(xml.decode())
    fp.close()

if __name__ == '__main__':
    fg_list = load_image.loadim('images')
    for fg in fg_list:
        bnd_info = generate_dict.object_dict(fg)
        fg = cv2.imread(fg, -1)
        bg = cv2.imread('background/Fabrik.jpg', -1)
        object_bndbox = image_overlay.overlap(bg, fg, bnd_info)
        xml_generator(object_bndbox)
        print(object_bndbox)