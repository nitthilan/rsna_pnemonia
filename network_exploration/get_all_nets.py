from keras.applications import resnet50
from keras.applications import mobilenetv2
from keras_squeezenet import SqueezeNet
from keras.applications import vgg19

from keras_applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.engine.input_layer import Input

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.models import Model
from keras import optimizers
from keras.losses import binary_crossentropy, mean_squared_error
from keras.backend import int_shape

import numpy as np

from os import listdir
from os.path import isfile, join
import os

import matplotlib.image as mpimg
import time


# SqueezeNet: https://github.com/rcmalli/keras-squeezenet/blob/master/examples/example_keras_squeezenet.ipynb
# https://keras.io/applications/

def get_all_nets(network_name, include_top=True):
	if(network_name=="ResNet50"):
		model = resnet50.ResNet50(weights='imagenet',
			include_top=include_top, input_shape=(224, 224, 3))
		# if(include_top==False):
		# 	model.pop()
	elif(network_name=="MobileNetV2"):
		model = mobilenetv2.MobileNetV2(weights='imagenet',
			include_top=include_top, input_shape=(224, 224, 3))
	elif(network_name=="VGG19"):
		model = vgg19.VGG19(weights='imagenet',
			include_top=include_top)
	elif(network_name=="SqueezeNet"):
		model = SqueezeNet(weights='imagenet',
		include_top=include_top)
		# if(include_top==False):
		# 	model.pop()
		# 	model.pop()
		# 	model.pop()
		# 	model.pop()
	return model

def preprocess_image(network_name, x):
	if(network_name=="ResNet50"):
		x = resnet50.preprocess_input(x)
	elif(network_name=="MobileNetV2"):
		x = mobilenetv2.preprocess_input(x)
	elif(network_name=="VGG19"):
		x = vgg19.preprocess_input(x)
	elif(network_name=="SqueezeNet"):
		x = imagenet_utils.preprocess_input(x)
	return x

def decodepred(network_name, preds):
	if(network_name=="ResNet50"):
		preds = resnet50.decode_predictions(preds, top=3)[0]
	elif(network_name=="MobileNetV2"):
		preds = mobilenetv2.decode_predictions(preds, top=3)[0]
	elif(network_name=="VGG19"):
		preds = vgg19.decode_predictions(preds, top=3)[0]
	elif(network_name=="SqueezeNet"):
		preds = imagenet_utils.decode_predictions(preds, top=3)[0]
	return x

def analyse_model(model):
	print("All functions ", dir(model))
	print("Summary model ", model.summary())
	print("Layer details ", dir(model.layers[2]))
	for i, layer in enumerate(model.layers):
		print("Length in each layer ", i, layer.name,
			layer.input_shape, layer.output_shape,
			len(layer.weights))
		if(len(layer.weights)):
			for j, weight in enumerate(layer.weights):
				print("Weights ", j, weight.shape)
	return

def customLoss(y_true, y_pred):
	box_true = y_true[:,0]
	prob_true = y_true[:,1:]
	box_pred = y_pred[:,0]
	prob_pred = y_pred[:,1:]
	print("Loss structure ", int_shape(y_true), int_shape(y_pred))
	prob_loss = K.binary_crossentropy(prob_true, prob_pred)
	box_loss = K.mean(K.square(box_true - box_pred), axis=-1)
	return K.mean(prob_loss*box_loss, axis=-1)

def add_classifier(base_model):

	for layer in base_model.layers:
		layer.trainable = False
		
	x = base_model.output
	x = Flatten()(x)
	x = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(512, kernel_regularizer=regularizers.l2(0.01))(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	prob = Dense(1, activation='sigmoid', name='prob_out')(x)
	box = Dense(4, activation='relu', name='box_out')(x)

	# outputs = concatenate([prob, box])
	# model = Model(inputs = base_model.input, outputs = outputs)
	
	model = Model(inputs = base_model.input, outputs = [prob, box])

	# initiate RMSprop optimizer
	opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(
		loss={'prob_out': 'binary_crossentropy', 'box_out': 'mean_squared_error'},
		loss_weights=[1.0, 1.0/(1024*4)],
		# loss = customLoss,
	    optimizer=opt)
	return model

def get_all_prediction(image_filelist):
	prediction_list = []
	for filename in image_filelist:

		# img = image.load_img(os.path.join(imagenet_path, filename), target_size=(224, 224))
		img = image.load_img(os.path.join(imagenet_path, filename), target_size=(227, 227)) # Squeezenet
		# img1 = mpimg.imread(os.path.join(imagenet_path, filename))
		# print(img1.shape)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = imagenet_utils.preprocess_input(x)

		preds = model.predict(x)
		# decode the results into a list of tuples (class, description, probability)
		# (one such list for each sample in the batch)
		print('Predicted:', filename, imagenet_utils.decode_predictions(preds, top=3)[0])
		print("Pred values ", np.argmax(preds))
		# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
		prediction_list.append(preds)
	return prediction_list

# image_array, filename_list = load_images(df_test_preprocess_filename)

# print("image dimensions ", image_array.shape)
# network_types_list = ["ResNet50", "MobileNetV2", "VGG19"] # , "SqueezeNet"
# for network_type in network_types_list:
# 	print("Network Type ", network_type)
# 	model = get_all_nets(network_type, include_top=False)
# 	analyse_model(model)
# 	model = add_classifier(model)

# imagenet_path = "/mnt/additional/aryan/imagenet_validation_data/ILSVRC2012_img_val/"
# # http://www.image-net.org/challenges/LSVRC/2012/
# # https://cv-tricks.com/tensorflow-tutorial/keras/
# # Finding actual predictions
# # http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

# image_filelist = [f for f in listdir(imagenet_path) if isfile(join(imagenet_path, f))]

# print("Number of files ", len(image_filelist))


# start_time = time.time()
# get_all_prediction(image_filelist[:10])
# total_time = time.time() - start_time
# print("Total prediction time ", total_time)

# print("File list ", image_filelist[:10]) 