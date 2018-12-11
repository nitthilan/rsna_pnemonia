import keras
import pandas as pd
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import csv


from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

import os
import pickle
import numpy as np

import get_all_nets as gan
import get_conv_net as gcn
import training as tr

base_folder = "/mnt/additional/nitthilan/data/kaggle/"
df_train_label = base_folder + 'stage_1_train_labels.csv'

df_preprocess_base_folder = base_folder + "preprocess_folder/"
df_train_preprocess_filename = df_preprocess_base_folder + "train_224.npz"
df_test_preprocess_filename = df_preprocess_base_folder + "test_224.npz"


def dump_csv(pred_list_0, pred_list_1, filelist, filename):
	with open(filename, 'w') as csvfile:
		fieldnames = ['patientId', 'PredictionString']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for pred0, pred1, file in zip(pred_list_0, pred_list_1, filelist):
			pred_output = {
				'patientId': str(file)[2:-5], 
				'PredictionString': str(pred0[0])+" "+\
					str(int(pred1[0]))+" "+ \
					str(int(pred1[1]))+" "+ \
					str(int(pred1[2]))+" "+ \
					str(int(pred1[3])),
			}
			writer.writerow(pred_output)

	return

# train_image_list, train_filename_list = \
#   tr.load_images(df_train_preprocess_filename)
# train_name_idx_map = tr.get_inv_mapping(train_filename_list)
test_image_list, test_filename_list = \
  tr.load_images(df_test_preprocess_filename)
test_name_idx_map = tr.get_inv_mapping(test_filename_list)

network_name = "ResNet50"#"MobileNetV2"
save_dir = os.path.join(base_folder, 'saved_models')
weight_path = os.path.join(save_dir, \
        "keras_"+network_name+"_weight_.h5")

test_image_list = np.expand_dims(test_image_list, axis=3)
test_image_list = np.repeat(test_image_list, 3, axis=3)
test_image_list_np = gan.preprocess_image(network_name, test_image_list.astype(float))

with tf.device("/gpu:0"):
  # load the weights and model from .h5
  model = load_model(weight_path)
print(weight_path)
model.summary()

test_pred = model.predict(test_image_list_np, verbose=1)
print("Pred output ", test_pred[0].shape, test_pred[1].shape)
dump_csv(test_pred[0], test_pred[1], test_filename_list, os.path.join(base_folder, "test_output.csv"))