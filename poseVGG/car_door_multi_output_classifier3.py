'''
Multi-Output Classification for car door orientation
ResNet backend
@author: Hang Wu
@date: 2018.11.21

USAGE
python car_door_multi_output_classifier2.py --dataset /home/hangwu/Workspace/Car_Door\
    --model output/pose.model \
    --label /home/hangwu/Workspace/car_door_pose_half.json \
    --latitudebin output/latitude_lb.pickle \
    --longitudebin output/longitude_lb.pickle


'''
import matplotlib
matplotlib.use("Agg")
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from posenet2 import PoseNet
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import random
import json
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", 
	default="/home/hangwu/Workspace/car_door_square",
	# required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", 
	default="output/pose.model",
	# required=True,
	help="path to output model")
ap.add_argument("-l", "--label", 
	default="/home/hangwu/Workspace/annotations/car_door_pose_half.json",
	# required=True, 
    help="path to annotation .json file")
ap.add_argument("-a", "--latitudebin", 
	default="output/latitude_lb.pickle",
	# required=True,
	help="path to output latitude label binarizer")
ap.add_argument("-o", "--longitudebin", 
	default="output/longitude_lb.pickle",
	# required=True,
	help="path to output longitude label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot",
	help="base filename for generated plots")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

EPOCHS = 300
INIT_LR = 1e-4
BS = 128
IMAGE_DIMS = (128, 128, 3)

# initialize the data, latitude value (0~59) and longitude value(0~359)
data = []
latitudeLabels = []
longitudeLabels = []

# loop over the input images

if not os.path.exists(args["label"]):
    raise ValueError('`annotation_path` does not exist.')
    
annotation_json = open(args["label"], 'r')
annotation_list = json.load(annotation_json)
#### pruefen >>>>
_ = 0
#### pruefen <<<<
for d in annotation_list['annotations']:
    image_name = d.get('image_id')
    latitude = d.get('latitude')
    longitude = d.get('longitude')

    # Bounding Box information >>>
    # xmin = d.get('xmin')
    # ymin = d.get('ymin')
    # xmax = d.get('xmax')
    # ymax = d.get('ymax')
    # bbox_width = xmax - xmin
    # bbox_height = ymax - ymin
    # print(bbox_height)
    # print(bbox_width)
    # side_len_diff_half = round(abs(bbox_height - bbox_width) / 2)
    # print(side_len_diff_half)
    # Bounding Box information <<<
    if args["dataset"] is not None:
        image_name = os.path.join(args["dataset"], image_name)
    image = cv2.imread(image_name)
	# Crop image
    # crop_image = image[ymin:ymax, xmin:xmax]
    """
	fill the short side to get a square >>>>
	"""
    # if bbox_height >= bbox_width:
    # 	new_patch = np.zeros((bbox_height, bbox_height ,3), np.uint8)
    # 	for row in range(crop_image.shape[0]):
    # 		for col in range(crop_image.shape[1]):
    # 			new_patch[row, col + side_len_diff_half] = crop_image[row, col]
    # else:
    # 	new_patch = np.zeros((bbox_width, bbox_width ,3), np.uint8)
    # 	for row in range(crop_image.shape[0]):
    # 		for col in range(crop_image.shape[1]):
    # 			new_patch[row + side_len_diff_half, col] = crop_image[row, col]
    """
	fill the short side to get a square <<<<
	"""
    # cv2.imshow("patch1", new_patch)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # cv2.imshow("patch2", new_patch)
    # cv2.imshow("crop image", crop_image)
    # cv2.imshow("Test", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
	# test >>>
    # cv2.imshow("resized crop_image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
	# test <<<
	
    image = img_to_array(image)
    data.append(image)
    latitudeLabels.append(latitude)
    longitudeLabels.append(longitude)
    #### pruefen >>>>
    if _ % 100 == 0:
        print('{:.2f}% finished'.format(_/len(annotation_list['annotations'])*100))
    _ += 1
    #### pruefen <<<<

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(annotation_list['annotations']), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
latitudeLabels = np.array(latitudeLabels)
longitudeLabels = np.array(longitudeLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
latitudeLB = LabelBinarizer()
longitudeLB = LabelBinarizer()
latitudeLabels = latitudeLB.fit_transform(latitudeLabels)
longitudeLabels = longitudeLB.fit_transform(longitudeLabels)


# save the latitude binarizer to disk
print("[INFO] serializing latitude label binarizer...")
f = open(args["latitudebin"], "wb")
f.write(pickle.dumps(latitudeLB))
f.close()

# save the longitude binarizer to disk
print("[INFO] serializing longitude label binarizer...")
f = open(args["longitudebin"], "wb")
f.write(pickle.dumps(longitudeLB))
f.close()


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, latitudeLabels, longitudeLabels,
            test_size=0.2)
(trainX, testX, trainLatitudeY, testLatitudeY, 
    trainLongitudeY, testLongitudeY) = split



# initialize VGG multi-output network

model = PoseNet.VGG16_mod(128,128,
        numLatitudes=len(latitudeLB.classes_),
        numLongitudes=len(longitudeLB.classes_),
        finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"latitude_output": "categorical_crossentropy",
	"longitude_output": "categorical_crossentropy",
}
lossWeights = {"latitude_output": 1.0, "longitude_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(
	# data,
	# {"latitude_output": latitudeLabels, "longitude_output": longitudeLabels},
	trainX,
	{"latitude_output": trainLatitudeY, "longitude_output": trainLongitudeY},
	validation_data=(testX,
		{"latitude_output": testLatitudeY, "longitude_output": testLongitudeY}),
	epochs=EPOCHS,
	verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the total loss, latitude loss, and longitude loss
lossNames = ["loss", "latitude_output_loss", "longitude_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()


# save the losses figure
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()


# create a new figure for the accuracies
accuracyNames = ["latitude_output_acc", "longitude_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))


# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()





