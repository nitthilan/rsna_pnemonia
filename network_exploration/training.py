'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import pandas as pd
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint


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




base_folder = "/mnt/additional/nitthilan/data/kaggle/"
df_train_label = base_folder + 'stage_1_train_labels.csv'
df_train_class_info = base_folder + 'stage_1_detailed_class_info.csv'
df_train_base_folder = base_folder + "stage_1_train_images/"
df_test_base_folder = base_folder + "stage_1_test_images/"

df_preprocess_base_folder = base_folder + "preprocess_folder/"
df_train_preprocess_filename = df_preprocess_base_folder + "train_224.npz"
df_test_preprocess_filename = df_preprocess_base_folder + "test_224.npz"


def load_images(filename):
  filevalue = np.load(filename)
  # print("FileValue ", list(filevalue.keys()), filevalue["filename_list"])
  return filevalue["image_array"], filevalue["filename_list"]

def get_inv_mapping(filename_list):
  name_idx_map = {}
  for idx, file in enumerate(filename_list):
    name_idx_map[file.decode('ascii')] = idx
    # print(file.decode('ascii'), idx)
  # print("Name Idx map ", name_idx_map, filename_list)
  return name_idx_map

def get_image_list(filename_list, name_idx_map, image_list):
  image_list = []
  for file in filename_list:
    image_list.append(image_list[name_idx_map[file]])
  return np.array(image_list)

def get_train_info(filename):
  dftl = pd.read_csv(filename)
  dftl_np = dftl.values
  x_list = dftl_np[:,1]
  y_list = dftl_np[:,2]
  width_list = dftl_np[:,3]
  hgt_list = dftl_np[:,4]
  pred_list = dftl_np[:,5]
  file_list = dftl_np[:,0]
  return file_list, x_list, y_list, width_list, hgt_list, pred_list

def get_train_data(train_image_list, 
  train_name_idx_map, train_label_filename, is_repeat):
  file_list, x_list, y_list, width_list, hgt_list, pred_list \
    = get_train_info(train_label_filename)

  image_list = []
  pred_val_list = []
  box_list = []
  numbox_list = []
  total_files = len(file_list)
  i = 0
  while i != total_files:
    # print("List index ", i)
    cur_file = file_list[i]
    image = train_image_list[train_name_idx_map[cur_file+".dcm"]]
    image = np.expand_dims(image, axis=2)
    if(is_repeat):
      image = np.repeat(image, 3, axis=2)
    # print("Shape ", image.shape)
    image_list.append(image)
    pred_val_list.append(pred_list[i])
    box_list.append(np.zeros((4, 4)))
    if(pred_list[i]):
      j = 0
      while (file_list[i+j] == cur_file):
        box_list[-1][j, 0] = x_list[i+j]
        box_list[-1][j, 1] = y_list[i+j]
        box_list[-1][j, 2] = width_list[i+j]
        box_list[-1][j, 3] = hgt_list[i+j]
        j+=1
      i+=j
      numbox_list.append(j)
    else:
      numbox_list.append(0)
      i+=1
  image_list_np = np.array(image_list)
  pred_val_list_np = np.array(pred_val_list)
  box_list_np = np.array(box_list)
  numbox_list_np = np.array(numbox_list)
  return (image_list_np, pred_val_list_np, box_list_np, numbox_list_np)

def concatenate_output(pred_list_np, box_list_np):
  y = []
  for pred, box in zip(pred_list_np, box_list_np):
    pred_box = np.zeros(5)
    pred_box[0] = pred
    pred_box[1:] = box[0]
    y.append(pred_box)
  return np.array(y)

def get_non_zero_pred(pred_list_np, box_list_np, image_list_np):
  box_list = []
  image_list = []
  pred_list = []
  for i,pred in enumerate(pred_list_np):
    if(pred):
      box_list.append(box_list_np[i])
      image_list.append(image_list_np[i])
      pred_list.append(pred)
  print("The total boxes ", len(pred_list))
  return np.array(pred_list), np.array(box_list), np.array(image_list)



# gan.analyse_model(model)

if __name__ == "__main__":
  train_image_list, train_filename_list = \
    load_images(df_train_preprocess_filename)
  train_name_idx_map = get_inv_mapping(train_filename_list)
  test_image_list, test_filename_list = \
    load_images(df_test_preprocess_filename)
  test_name_idx_map = get_inv_mapping(test_filename_list)

  (image_list_np, pred_list_np, box_list_np, numbox_list_np) = \
    get_train_data(train_image_list, 
      train_name_idx_map, df_train_label, True)


  # (image_list_np, pred_list_np, box_list_np, numbox_list_np) = \
  #   get_train_data(train_image_list, 
  #     train_name_idx_map, df_train_label, False)


  # print("Train label info ", train_label_list)

  total_images = image_list_np.shape[0]
  # image_list_np = np.expand_dims(image_list_np, axis=3)
  print("Train image list ", image_list_np.shape, pred_list_np.shape)

  network_name = "ResNet50"#"MobileNetV2"
  with tf.device('/gpu:1'):
    model = gan.get_all_nets(network_name, include_top=False)
    model = gan.add_classifier(model)
  image_list_np = gan.preprocess_image(network_name, image_list_np.astype(float))

  # model = gcn.get_conv_net(image_list_np.shape[1:], 1, 2, wgt_fname=None)

  num_train = int(total_images*0.8)

  # y = concatenate_output(pred_list_np, box_list_np)

  x_train = image_list_np[:num_train]
  x_val = image_list_np[num_train:]
  # y_train = y[:num_train]
  # y_val = y[num_train:]

  y_pred_train = pred_list_np[:num_train]
  y_pred_val = pred_list_np[num_train:]
  y_box_train = box_list_np[:num_train, 0, :]
  y_box_val = box_list_np[num_train:, 0, :]


  pred_list_np, box_list_np, image_list_np = \
    get_non_zero_pred(pred_list_np, box_list_np, image_list_np)


  save_dir = os.path.join(base_folder, 'saved_models')
  weight_path = os.path.join(save_dir, \
          "keras_"+network_name+"_weight_.h5")
  # model.save(weight_path)

  modelCheckpoint = ModelCheckpoint(weight_path, 
    monitor='loss', verbose=0, save_best_only=True, 
    save_weights_only=False, mode='auto', period=1)

  callbacks = [
              modelCheckpoint
              #   earlyStopping, 
              #   reduceonplateau,
              #   csv_logger
              ]



  batch_size = 32 #128
  epochs = 200


  print('Not using data augmentation.')
  model.fit(
            # x_train, y_train,
            # validation_data=(x_val, y_val),
            x_train, [y_pred_train, y_box_train],
            validation_data=(x_val, [y_pred_val, y_box_val]),
            batch_size=batch_size,
            epochs=epochs,
            callbacks = callbacks,
            shuffle=True)
  print('Saved trained model and weights at %s ' % weight_path)
