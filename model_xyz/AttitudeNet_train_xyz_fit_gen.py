"""
Multi-Output Classification for car door pose estimation
VGG16 backend
@author: Hang Wu
@date: 2018.11.21

USAGE
python car_door_multi_output_classifier2.py --dataset /home/hangwu/Workspace/Car_Door\
    --model output/pose.model \
    --label /home/hangwu/Workspace/car_door_pose_half.json \
    --latitudebin output/latitude_lb.pickle \
    --longitudebin output/longitude_lb.pickle

"""
import random

import matplotlib

matplotlib.use("Agg")
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from AttitudeNet_model_xyz import AttitudeNet
import matplotlib.pyplot as plt
import argparse
import time
import json
import pickle
import cv2
import re
import os

# used model for save_name, "mobilenet", "resnet", "vgg16"
model_name = "vgg16"

# construct the argument parse and parse the arguments
# =======================================================================
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
                default="bigdata/cropped",
                # required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model",
                default="bigdata/output/pose_{}.model".format(model_name),
                # required=True,
                help="path to output model")

ap.add_argument("-x", "--xLabelBin",
                default="bigdata/output/x_lb.pickle".format(model_name),
                # required=True,
                help="path to output x label binarizer")
ap.add_argument("-y", "--yLabelBin",
                default="bigdata/output/y_lb.pickle".format(model_name),
                # required=True,
                help="path to output y label binarizer")
ap.add_argument("-z", "--zLabelBin",
                default="bigdata/output/z_lb.pickle".format(model_name),
                # required=True,
                help="path to output z label binarizer")

ap.add_argument("-p", "--plot", type=str, default="plot",
                help="base filename for generated plots")
args = vars(ap.parse_args())
# =======================================================================

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

EPOCHS = 2000
INIT_LR = 1e-5
BS = 128
IMAGE_DIMS = (224, 224, 3)

# initialize the data, latitude value (0~59) and longitude value(0~359)
data = []
xLabels = []
yLabels = []
zLabels = []


def loadim(image_path, ext='png'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext):
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path, filename)
            image_list.append(image_abs_path)
    return image_list


imList = loadim(args["dataset"])

# to show the process
_ = 0

for p in imList:
    name = p.split(os.path.sep)[-1]
    match = re.match(
        r'([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(_+)([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(_+)([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(\.png)',
        name, re.I)
    xLabels.append(match.groups()[2])
    yLabels.append(match.groups()[6])
    zLabels.append(match.groups()[10])
    # process >>>>
    if _ % 100 == 0:
        print('{:.2f}% finished'.format(_ / len(imList) * 100))
    _ += 1
    # process <<<<
print(len(set(xLabels)))
print(len(set(yLabels)))
print(len(set(zLabels)))
print(xLabels.index('6'))
print(xLabels[:10])
print(xLabels[8])

# convert the label lists to NumPy arrays prior to binarization
xLabels_np = np.array(xLabels)
yLabels_np = np.array(yLabels)
zLabels_np = np.array(zLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
xLabelLB = LabelBinarizer()
yLabelLB = LabelBinarizer()
zLabelLB = LabelBinarizer()
print(xLabels_np)

xLabels_oh = xLabelLB.fit_transform(xLabels_np)
yLabels_oh = yLabelLB.fit_transform(yLabels_np)
zLabels_oh = zLabelLB.fit_transform(zLabels_np)

print(len(xLabels_oh[8]))
print()

# save the x label binarizer to disk
print("[INFO] serializing x label binarizer...")
f = open(args["xLabelBin"], "wb")
f.write(pickle.dumps(xLabelLB))
f.close()

# save the y label binarizer to disk
print("[INFO] serializing y label binarizer...")
f = open(args["yLabelBin"], "wb")
f.write(pickle.dumps(yLabelLB))
f.close()

# save the z label binarizer to disk
print("[INFO] serializing z label binarizer...")
f = open(args["zLabelBin"], "wb")
f.write(pickle.dumps(zLabelLB))
f.close()


def load_data(img_list, part, shuffle):
    if shuffle:
        random.shuffle(img_list)
    n = int(len(img_list) * part)
    short_list = img_list[:n]
    while True:
        p = random.choice(short_list)
        image = cv2.imread(p)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)

        # scale the raw pixel intensities to the range [0, 1] and convert to
        # a NumPy array
        image = np.array(image, dtype="float") / 255.0

        name = p.split(os.path.sep)[-1]
        match = re.match(
            r'([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(_+)([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(_+)([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(\.png)',
            name, re.I)
        x = match.groups()[2]
        y = match.groups()[6]
        z = match.groups()[10]

        x_index = xLabels.index(x)
        y_index = yLabels.index(y)
        z_index = zLabels.index(z)

        x_oh = xLabels_oh[x_index]
        y_oh = yLabels_oh[y_index]
        z_oh = zLabels_oh[z_index]

        yield (np.expand_dims(image, axis=0),
               {"x_label_output": np.expand_dims(x_oh, axis=0),
                "y_label_output": np.expand_dims(y_oh, axis=0),
                "z_label_output": np.expand_dims(z_oh, axis=0)}
               )


a = load_data(imList, 1, True)
print(a.__next__())

# exit()
print("[INFO] VGG16")
model = AttitudeNet.VGG16_mod(224, 224,
                              len(xLabelLB.classes_),
                              len(yLabelLB.classes_),
                              len(zLabelLB.classes_))

# define three dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
    "x_label_output": "categorical_crossentropy",
    "y_label_output": "categorical_crossentropy",
    "z_label_output": "categorical_crossentropy",
}
lossWeights = {"x_label_output": 2.0, "y_label_output": 1.0, "z_label_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
              metrics=["accuracy"])

start_train = time.time()

H = model.fit_generator(
    generator=load_data(imList, 1, True),
    steps_per_epoch=64,
    epochs=EPOCHS,
    verbose=1,
    callbacks=None,
    validation_data=load_data(imList, 0.2, True),
    validation_steps=64,
    class_weight=None,
    max_queue_size=10,
    workers=8,
    use_multiprocessing=True,
    shuffle=True,
    initial_epoch=0
)

end_train = time.time()
hours = round((end_train - start_train) / 60 / 60, 2)
print(f'Training took {hours} minutes')

# for p in imList:
#     image = cv2.imread(p)
#     image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
#     image = img_to_array(image)
#     data.append(image)

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
# data = np.array(data, dtype="float") / 255.0
# print("[INFO] data matrix: {} images ({:.2f}MB)".format(
#     len(imList), data.nbytes / (1024 * 1000.0)))


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

# split = train_test_split(data, xLabels_oh, yLabels_oh, zLabels_oh,
#                          test_size=0.2)
# (trainX, testX,
#  trainXLabelY, testXLabelY,
#  trainYLabelY, testYLabelY,
#  trainZLabelY, testZLabelY) = split
"""
start_train = time.time()
# train the network to perform multi-output classification
H = model.fit(
    # data,
    # {"latitude_output": latitudeLabels, "longitude_output": longitudeLabels},
    data,
    {"x_label_output": xLabels, "y_label_output": yLabels, "z_label_output": zLabels},
    validation_data=(testX,
                     {"x_label_output": testXLabelY,
                      "y_label_output": testYLabelY,
                      "z_label_output": testZLabelY}),
    epochs=EPOCHS,
    verbose=1)

end_train = time.time()
hours = round((end_train - start_train) / 60 / 60, 2)
print(f'Training took {hours} minutes')
"""
# save the loss/accuracy history to file
with open('bigdata/output/history_{}.json'.format(model_name), 'w') as outfile:
    json.dump(H.history, outfile)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the total loss, latitude loss, and longitude loss
lossNames = ["loss", "x_label_output_loss", "y_label_output_loss", "z_label_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))

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
# ======================= Diagramm Path ===============================================
plt.savefig("bigdata/output/all_{}_losses_{}.pdf".format(args["plot"], model_name))
plt.close()

# create a new figure for the accuracies
accuracyNames = ["x_label_output_acc", "y_label_output_acc", "z_label_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(8, 8))

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
# ======================= Diagramm Path ===============================================
plt.savefig("bigdata/output/all_{}_accs_{}.pdf".format(args["plot"], model_name))
plt.close()
