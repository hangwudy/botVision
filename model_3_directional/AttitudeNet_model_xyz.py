# import the necessary packages
from keras import applications
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf


class AttitudeNet:
    # Test Model 1, ResNet-50
    @staticmethod
    def ResNet_mod(width, height, numLatitudes, numLongitudes,
                   finalAct="softmax"):
        base_model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(width, height, 3))
        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        x_la = Dense(numLatitudes)(x)
        latitudeBranch = Activation(finalAct, name="latitude_output")(x_la)

        x_lo = Dense(numLongitudes)(x)
        longitudeBranch = Activation(finalAct, name="longitude_output")(x_lo)

        model = Model(
            inputs=base_model.input,
            outputs=[latitudeBranch, longitudeBranch],
            name='attitudenet')

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        for layer in model.layers[:30]:
            layer.trainable = False

        return model

    # Test Model 2, VGG-16
    @staticmethod
    def VGG16_mod(width, height, numXLabels, numYLabels, numZLabels,
                  finalAct="softmax"):
        base_model = applications.VGG16(weights=None, include_top=False, input_shape=(width, height, 3))
        x = base_model.output

        x = Flatten()(x)

        x_x_label = Dense(256, activation='relu', name='fc1_1')(x)
        x_x_label = Dense(256, activation='relu', name='fc2_1')(x_x_label)
        x_x_label = Dense(numXLabels)(x_x_label)
        xLabelBranch = Activation(finalAct, name="x_label_output")(x_x_label)

        x_y_label = Dense(256, activation='relu', name='fc1_2')(x)
        x_y_label = Dense(256, activation='relu', name='fc2_2')(x_y_label)
        x_y_label = Dense(numYLabels)(x_y_label)
        yLabelBranch = Activation(finalAct, name="y_label_output")(x_y_label)

        x_z_label = Dense(256, activation='relu', name='fc1_3')(x)
        x_z_label = Dense(256, activation='relu', name='fc2_3')(x_z_label)
        x_z_label = Dense(numZLabels)(x_z_label)
        zLabelBranch = Activation(finalAct, name="z_label_output")(x_z_label)

        model = Model(
            inputs=base_model.input,
            outputs=[xLabelBranch, yLabelBranch, zLabelBranch],
            name='attitudenet')

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        for layer in model.layers[:10]:
            layer.trainable = False

        return model
