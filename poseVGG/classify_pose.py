"""

# USAGE

python classify_pose.py --model output/pose.model \
    --latitudebin output/latitude_lb.pickle --longitudebin output/longitude_lb.pickle \
    --image /home/hangwu/Workspace/Car_Door


"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time
from numpy import random

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
                default="/home/hangwu/Workspace/multi-output-classification/output/pose.model",
                # required=True,
                help="path to trained model model")
ap.add_argument("-a", "--latitudebin",
                default="/home/hangwu/Workspace/multi-output-classification/output/latitude_lb.pickle",
                # required=True,
                help="path to output latitude label binarizer")
ap.add_argument("-o", "--longitudebin",
                default="/home/hangwu/Workspace/multi-output-classification/output/longitude_lb.pickle",
                # required=True,
                help="path to output longitude label binarizer")
ap.add_argument("-i", "--image",
                default="/home/hangwu/Workspace/car_door_half",
                # required=True,
                help="path to input image directory")
args = vars(ap.parse_args())

# load the trained convolutional neural network from disk, followed
# by the latitude and longitude label binarizers, respectively
print("[INFO] loading network...")
model = load_model(args["model"], custom_objects={"tf": tf})
latitudeLB = pickle.loads(open(args["latitudebin"], "rb").read())
longitudeLB = pickle.loads(open(args["longitudebin"], "rb").read())


def inference(image_path):
    # load the image
    image = cv2.imread(image_path)

    if image.shape[0] > image.shape[1]:
        img_width = image.shape[1]
        img_height = image.shape[0]
        side_len_diff_half = round(abs(img_height - img_width) / 2)
        new_patch = np.zeros((img_height, img_height, 3), np.uint8)
        for row in range(img_height):
            for col in range(img_width):
                new_patch[row, col + side_len_diff_half] = image[row, col]
        image = new_patch
        cv2.imshow("resized image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif image.shape[0] < image.shape[1]:
        img_width = image.shape[1]
        img_height = image.shape[0]
        side_len_diff_half = round(abs(img_height - img_width) / 2)
        new_patch = np.zeros((img_width, img_width, 3), np.uint8)
        for row in range(img_height):
            for col in range(img_width):
                new_patch[row + side_len_diff_half, col] = image[row, col]
        image = new_patch
        cv2.imshow("resized image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    output = imutils.resize(image, width=400)

    # pre-process the image for classification
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image using Keras' multi-output functionality
    print("[INFO] classifying image...")
    (latitudeProba, longitudeProba) = model.predict(image)

    # find indexes of both the latitude and longitude outputs with the
    # largest probabilities, then determine the corresponding class
    # labels
    latitudeIdx = latitudeProba[0].argmax()
    longitudeIdx = longitudeProba[0].argmax()
    latitudeLabel = latitudeLB.classes_[latitudeIdx]
    longitudeLabel = longitudeLB.classes_[longitudeIdx]

    # draw the latitude label and longitude label on the image
    latitudeText = "latitude: {} ({:.2f}%)".format(latitudeLabel,
                                                   latitudeProba[0][latitudeIdx] * 100)
    longitudeText = "longitude: {} ({:.2f}%)".format(longitudeLabel,
                                                     longitudeProba[0][longitudeIdx] * 100)
    cv2.putText(output, latitudeText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(output, longitudeText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    print("[INFO] {}".format(latitudeText))
    print("[INFO] {}".format(longitudeText))

    # show the output image
    cv2.imshow("Output", output)
    image_compare_name = 'car_door_{}_{}.png'.format(latitudeLabel, longitudeLabel)
    image_compare_path = os.path.join('/home/hangwu/Workspace/Car_Door', image_compare_name)
    print(image_compare_path)
    image_compare = cv2.imread(image_compare_path)
    image_compare = imutils.resize(image_compare, width=400)

    image_horizontal = np.hstack((output, image_compare))
    cv2.imshow("Comparison", image_compare)
    cv2.imshow("Reality and Prediction", image_horizontal)
    cv2.imwrite("Result_{:4.0f}.png".format(time.time()), image_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loadim(image_path='Car_Door', ext='png', key_word='car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path, filename)
            image_list.append(image_abs_path)
    return image_list


def main():
    # img_path_list = loadim(args["image"])
    # # print(img_path_list)
    # test_img = random.choice(img_path_list, 6)
    # # print(img_path_choice)

    test_img_1 = '/home/hangwu/Workspace/multi-output-classification/test_pics/car_door_test_1.png'
    test_img_2 = '/home/hangwu/Workspace/multi-output-classification/test_pics/car_door_cvt.png'
    test_img_3 = '/home/hangwu/Workspace/multi-output-classification/test_pics/car_door_cvt_2.png'
    test_img_4 = '/home/hangwu/Workspace/multi-output-classification/test_pics/car_door_test_2.png'
    test_img_5 = '/home/hangwu/Workspace/multi-output-classification/test_pics/door333.png'
    test_img = []
    test_img.append(test_img_1)
    test_img.append(test_img_2)
    test_img.append(test_img_3)
    test_img.append(test_img_4)
    test_img.append(test_img_5)

    for image_file in test_img:
        print(image_file)
        inference(image_file)


if __name__ == "__main__":
    main()
