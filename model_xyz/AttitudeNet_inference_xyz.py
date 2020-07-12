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
from numpy import random

# Model names, "vgg16", "mobilenet", "resnet"
model_name = "vgg16"
IMAGE_DIMS = (224, 224, 3)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
                default="xyz/render/output/pose_{}.model".format(model_name),
                # required=True,
                help="path to trained model model")

ap.add_argument("-x", "--xLabelBin",
                default="xyz/render/output/x_lb.pickle".format(model_name),
                # required=True,
                help="path to output x label binarizer")
ap.add_argument("-y", "--yLabelBin",
                default="xyz/render/output/y_lb.pickle".format(model_name),
                # required=True,
                help="path to output y label binarizer")
ap.add_argument("-z", "--zLabelBin",
                default="xyz/render/output/z_lb.pickle".format(model_name),
                # required=True,
                help="path to output z label binarizer")

ap.add_argument("-i", "--image",
                default="xyz/render/cropped",
                # required=True,
                help="path to input dataset (i.e., directory of images)")

ap.add_argument("-r", "--renderings",
                default="/home/wu/CyMePro/Dataset/renderings_square",
                # required=True,
                help="path to input image directory")
args = vars(ap.parse_args())

# load the trained convolutional neural network from disk, followed
# by the latitude and longitude label binarizers, respectively
print("[INFO] loading network...")
model = load_model(args["model"], custom_objects={"tf": tf})
xLabelLB = pickle.loads(open(args["xLabelBin"], "rb").read())
yLabelLB = pickle.loads(open(args["yLabelBin"], "rb").read())
zLabelLB = pickle.loads(open(args["zLabelBin"], "rb").read())


def inference(image_path):
    # load the image
    image = cv2.imread(image_path)

    # for images that are cropped according to the bounding box
    # Pad the rectangle images into squares
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
        # cv2.imshow("resized image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # For display
    output = imutils.resize(image, width=400)

    # pre-process the image for classification
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image using Keras' multi-output functionality
    print("[INFO] classifying image...")
    (x_proba, y_proba, z_proba) = model.predict(image)

    # find indexes of both the latitude and longitude outputs with the
    # largest probabilities, then determine the corresponding class
    # labels
    x_idx = x_proba[0].argmax()
    y_idx = y_proba[0].argmax()
    z_idx = z_proba[0].argmax()
    xLabel = xLabelLB.classes_[x_idx]
    yLabel = yLabelLB.classes_[y_idx]
    zLabel = zLabelLB.classes_[z_idx]

    # draw the latitude label and longitude label on the image
    x_text = "x: {} ({:.2f}%)".format(xLabel, x_proba[0][x_idx] * 100)
    y_text = "y: {} ({:.2f}%)".format(yLabel, y_proba[0][y_idx] * 100)
    z_text = "y: {} ({:.2f}%)".format(zLabel, z_proba[0][z_idx] * 100)
    cv2.putText(output, x_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(output, y_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(output, z_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    print("[INFO] {}".format(x_text))
    print("[INFO] {}".format(y_text))
    print("[INFO] {}".format(z_text))

    # show the output image according to the predicted attitude
    # cv2.imshow("Output", output)
    # image_compare_name = 'car_door_{}_{}.png'.format(xLabel, yLabel)
    # image_compare_path = os.path.join(args["renderings"], image_compare_name)
    # print(image_compare_path)
    # image_compare = cv2.imread(image_compare_path)
    # image_compare = imutils.resize(image_compare, width=400)
    # cv2.imshow("Comparison", image_compare)
    # cv2.waitKey(0)


def loadim(image_path, ext='png'):
    image_list = []

    for filename in os.listdir(image_path):
        if filename.endswith(ext):
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path, filename)
            image_list.append(image_abs_path)
    return image_list


def main():
    img_path_list = loadim(args["image"])
    # print(img_path_list)
    test_img = random.choice(img_path_list, 6)
    # print(img_path_choice)

    for image_file in test_img:
        print(image_file)
        inference(image_file)


if __name__ == "__main__":
    main()
