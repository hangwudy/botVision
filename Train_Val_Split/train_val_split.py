# -*- coding: UTF-8 -*-
# 2018/10/26 by HANG WU

import numpy
import os
import random
import shutil

# get image absolut path
def loadim(image_path = 'images', ext = 'jpg', key_word = 'car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list


# Path
# Images
Main_Dataset_Path='/home/hangwu/CyMePro/data/images'
Train_Dataset_Path='/home/hangwu/CyMePro/data/dataset/train_data'
Val_Dataset_Path='/home/hangwu/CyMePro/data/dataset/val_data'
Test_Dataset_Path='/home/hangwu/CyMePro/data/dataset/test_data'


img_path_dict = {
    'Main': Main_Dataset_Path,
    'Train': Train_Dataset_Path,
    'Val': Val_Dataset_Path,
    'Test': Test_Dataset_Path
}

# Masks
Main_Mask_Path='/home/hangwu/CyMePro/data/annotations/masks'
Train_Mask_Path='/home/hangwu/CyMePro/data/annotations/train_mask'
Val_Mask_Path='/home/hangwu/CyMePro/data/annotations/val_mask'
Test_Mask_Path='/home/hangwu/CyMePro/data/annotations/test_mask'

mask_path_dict = {
    'Main': Main_Mask_Path,
    'Train': Train_Mask_Path,
    'Val': Val_Mask_Path,
    'Test': Test_Mask_Path
}

# XMLs
Main_XML_Path='/home/hangwu/CyMePro/data/annotations/xmls'
Train_XML_Path='/home/hangwu/CyMePro/data/annotations/train_xml'
Val_XML_Path='/home/hangwu/CyMePro/data/annotations/val_xml'
Test_XML_Path='/home/hangwu/CyMePro/data/annotations/test_xml'

xml_path_dict = {
    'Main': Main_XML_Path,
    'Train': Train_XML_Path,
    'Val': Val_XML_Path,
    'Test': Test_XML_Path
}



# training portion
TrainR=0.7
# validating portion
ValR=0.2
# total num
PreImNum=100
fileIdLen=6

# image id set
ImIdSet=loadim(Main_Dataset_Path)
# print(ImIdSet)

# shuffle the list
random.shuffle(ImIdSet)
ImNum=len(ImIdSet)



# training number
TrainNum=int(TrainR*ImNum)
# validating number
ValNum=int(ValR*ImNum)

# get the first TrainNum data
TrainImId=ImIdSet[:TrainNum-1]
TrainImId.sort()
# get the val data
ValImId=ImIdSet[TrainNum:TrainNum+ValNum-1]
ValImId.sort()
# train + val = trainval
TrainValImId=list(set(TrainImId).union(set(ValImId)))
TrainValImId.sort()
# get the test 
TestImId=(list(set(ImIdSet).difference(set(TrainValImId))))
TestImId.sort()

# combine the sets in dictionary
TrainValTestIds={}
TrainValTestIds['Train']=TrainImId
TrainValTestIds['Val']=ValImId
# TrainValTestIds['trainval']=TrainValImId
TrainValTestIds['Test']=TestImId



print(TrainValTestIds.keys())
for key in TrainValTestIds.keys():
    # print(key)
    for value in TrainValTestIds[key]:
        # print(value)
        # file name 
        file_name_without_ext = value.split(os.path.sep)[-1][:-4]
        # images
        img_dest_path = img_path_dict[key]
        img_dest_file = os.path.join(img_dest_path, file_name_without_ext+'.jpg')
        shutil.copy2(value, img_dest_file)
        # masks
        mask_origin_path = mask_path_dict['Main']
        mask_origin_file = os.path.join(mask_origin_path, file_name_without_ext+'.png')
        mask_dest_path = mask_path_dict[key]
        mask_dest_file = os.path.join(mask_dest_path, file_name_without_ext+'.png')
        shutil.copy2(mask_origin_file, mask_dest_file)
        # XMLs
        xml_origin_path = xml_path_dict['Main']
        xml_origin_file = os.path.join(xml_origin_path, file_name_without_ext+'.xml')
        xml_dest_path = xml_path_dict[key]
        xml_dest_file = os.path.join(xml_dest_path, file_name_without_ext+'.xml')
        shutil.copy2(xml_origin_file, xml_dest_file)

