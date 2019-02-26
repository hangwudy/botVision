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



class PoseNet:
	'''
	@staticmethod
	def build_shared_layers(inputs, numLongitudes,
		finalAct="softmax", chanDim=-1):
		# VGG16
		# Block 1
		x = Conv2D(64, (3, 3),
						activation='relu',
						padding='same',
						name='block1_conv1')(inputs)
		x = Conv2D(64, (3, 3),
						activation='relu',
						padding='same',
						name='block1_conv2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
		
		# Block 2
		x = Conv2D(128, (3, 3),
						activation='relu',
						padding='same',
						name='block2_conv1')(x)
		x = Conv2D(128, (3, 3),
						activation='relu',
						padding='same',
						name='block2_conv2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		# Block 3
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv1')(x)
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv2')(x)
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		# Block 4
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv1')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv3')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		# Block 5
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv1_2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv2_2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv3_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_2')(x)

		# return the longitude prediction sub-network
		return x
	
	@staticmethod
	def build_latitude_branch(inputs, numLatitudes,
		finalAct="softmax", chanDim=-1):
		# VGG16
		# Block 1
		x = Conv2D(64, (3, 3),
						activation='relu',
						padding='same',
						name='block1_conv1_1')(inputs)
		x = Conv2D(64, (3, 3),
						activation='relu',
						padding='same',
						name='block1_conv2_1')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_1')(x)
		
		# Block 2
		x = Conv2D(128, (3, 3),
						activation='relu',
						padding='same',
						name='block2_conv1_1')(x)
		x = Conv2D(128, (3, 3),
						activation='relu',
						padding='same',
						name='block2_conv2_1')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_1')(x)

		# Block 3
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv1_1')(x)
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv2_1')(x)
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv3_1')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_1')(x)

		# Block 4
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv1_1')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv2_1')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv3_1')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_1')(x)

		# Block 5
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv1_1')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv2_1')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv3_1')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_1')(x)

		# define a branch of output layers for the number of latitudes
		x = Flatten(name='flatten_1')(x)
		x = Dense(4096, activation='relu', name='fc1_1')(x)
		x = Dense(4096, activation='relu', name='fc2_1')(x)
		x = Dense(numLatitudes)(x)
		x = Activation(finalAct, name="latitude_output")(x)

		# return the latitude prediction sub-network
		return x

	@staticmethod
	def build_longitude_branch(inputs, numLongitudes,
		finalAct="softmax", chanDim=-1):
		# VGG16
		# Block 1
		x = Conv2D(64, (3, 3),
						activation='relu',
						padding='same',
						name='block1_conv1_2')(inputs)
		x = Conv2D(64, (3, 3),
						activation='relu',
						padding='same',
						name='block1_conv2_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_2')(x)
		
		# Block 2
		x = Conv2D(128, (3, 3),
						activation='relu',
						padding='same',
						name='block2_conv1_2')(x)
		x = Conv2D(128, (3, 3),
						activation='relu',
						padding='same',
						name='block2_conv2_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_2')(x)

		# Block 3
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv1_2')(x)
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv2_2')(x)
		x = Conv2D(256, (3, 3),
						activation='relu',
						padding='same',
						name='block3_conv3_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_2')(x)

		# Block 4
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv1_2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv2_2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block4_conv3_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_2')(x)

		# Block 5
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv1_2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv2_2')(x)
		x = Conv2D(512, (3, 3),
						activation='relu',
						padding='same',
						name='block5_conv3_2')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_2')(x)

		# define a branch of output layers for the number of longitudes
		x = Flatten(name='flatten_2')(x)
		x = Dense(4096, activation='relu', name='fc1_2')(x)
		x = Dense(4096, activation='relu', name='fc2_2')(x)
		x = Dense(numLongitudes)(x)
		x = Activation(finalAct, name="longitude_output")(x)

		# return the longitude prediction sub-network
		return x

	@staticmethod
	def build(width, height, numLatitudes, numLongitudes,
		finalAct="softmax"):
		# initialize the input shape and channel dimension (this code
		# assumes you are using TensorFlow which utilizes channels
		# last ordering)
		inputShape = (height, width, 3)
		chanDim = -1

		# construct both the "category" and "color" sub-networks
		inputs = Input(shape=inputShape)
		latitudeBranch = PoseNet.build_latitude_branch(inputs,
			numLatitudes, finalAct=finalAct, chanDim=chanDim)
		longitudeBranch = PoseNet.build_longitude_branch(inputs,
			numLongitudes, finalAct=finalAct, chanDim=chanDim)

		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the clothing category
		# branch and another for the color branch, respectively
		model = Model(
			inputs=inputs,
			outputs=[latitudeBranch, longitudeBranch],
			name="posenet")
		# weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
		# model.load_weights(weights_path)
		# return the constructed network architecture
		return model
		'''
	@staticmethod
	def ResNet_mod(width, height, numLatitudes, numLongitudes,
		finalAct="softmax"):
		base_model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(width, height, 3))
		x = base_model.output
		
		x = GlobalAveragePooling2D()(x)
		# x_la = Dense(128, activation='relu', name='fc1_1')(x)
		# x_la = Dense(1024, activation='relu', name='fc2_1')(x_la)
		x_la = Dense(numLatitudes)(x)
		latitudeBranch = Activation(finalAct, name="latitude_output")(x_la)

		# x_lo = Dense(128, activation='relu', name='fc1_2')(x)
		# x_lo = Dense(1024, activation='relu', name='fc2_2')(x_lo)
		x_lo = Dense(numLongitudes)(x)
		longitudeBranch = Activation(finalAct, name="longitude_output")(x_lo)

		model = Model(
						inputs=base_model.input,
						outputs=[latitudeBranch, longitudeBranch],
						name='posenet')

		for i,layer in enumerate(model.layers):
			print(i,layer.name)

		for layer in model.layers:
			layer.trainable = False

		return model

	@staticmethod
	def VGG16_mod(width, height, numLatitudes, numLongitudes,
		finalAct="softmax"):
		base_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(width, height, 3))
		x = base_model.output
		
		x = Flatten()(x)

		x_la = Dense(1024, activation='relu', name='fc1_1')(x)
		x_la = Dense(1024, activation='relu', name='fc2_1')(x_la)
		x_la = Dense(numLatitudes)(x_la)
		latitudeBranch = Activation(finalAct, name="latitude_output")(x_la)

		x_lo = Dense(1024, activation='relu', name='fc1_2')(x)
		x_lo = Dense(1024, activation='relu', name='fc2_2')(x_lo)
		x_lo = Dense(numLongitudes)(x_lo)
		longitudeBranch = Activation(finalAct, name="longitude_output")(x_lo)

		model = Model(
						inputs=base_model.input,
						outputs=[latitudeBranch, longitudeBranch],
						name='posenet')

		for i,layer in enumerate(model.layers):
			print(i,layer.name)

		for layer in model.layers[:10]:
			layer.trainable = False

		return model


	@staticmethod
	def MobileNet_mod(width, height, numLatitudes, numLongitudes,
		finalAct="softmax"):
		base_model = applications.MobileNet(alpha=1.0, depth_multiplier=1, 
											weights="imagenet", include_top=False, input_shape=(width, height, 3))
		x = base_model.output
		
		x = GlobalAveragePooling2D()(x)
		# x_lo = Flatten()(x)
		# x = GlobalAveragePooling2D()(x)

		# x_la = Dense(16, activation='relu', name='fc1_1')(x)
		# x_la = Dense(512, activation='relu', name='fc2_1')(x_la)
		x_la = Dense(numLatitudes)(x)
		latitudeBranch = Activation(finalAct, name="latitude_output")(x_la)

		# x_lo = Dense(64, activation='relu', name='fc1_2')(x)
		# x_lo = Dense(512, activation='relu', name='fc2_2')(x_lo)
		x_lo = Dense(numLongitudes)(x)
		longitudeBranch = Activation(finalAct, name="longitude_output")(x_lo)

		model = Model(
						inputs=base_model.input,
						outputs=[latitudeBranch, longitudeBranch],
						name='posenet')

		for i,layer in enumerate(model.layers):
			print(i,layer.name)

		for layer in model.layers[:20]:
			layer.trainable = False

		return model
