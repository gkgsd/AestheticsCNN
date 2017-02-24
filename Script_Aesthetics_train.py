from __future__ import division
import sys
sys.path.insert(0, '../../MyPackages/')
sys.path.insert(0, '../../MyPackages/Inception-v4/')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2, os, random, time
from tqdm import tqdm

## set global parameters: backend and image_dim_ordering for keras, and the gpu configuration
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # or 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from keras import backend as K
import tensorflow as tf

K.set_image_dim_ordering('tf')
## for tensorflow sesstion
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
K.set_session(sess)

print K.image_dim_ordering(), K.backend()

## import keras layers
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.utils.visualize_util import plot
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard

from inception_v4 import create_inception_v4
from Package_dataset import Load_AVAdataset
from Package_network import loop_batch


## load the AVA dataset, hdf5 format
h5f_AVA = Load_AVAdataset(target_size=(224,224))
images_train = h5f_AVA['images_train']
scores_train = h5f_AVA['scores_train']
scores_train = scores_train[:,0]
sample_weights = h5f_AVA['sample_weights_train'][:]

images_test_even = h5f_AVA['images_test_even']
scores_test_even = h5f_AVA['scores_test_even']
scores_test_even = scores_test_even[:,0]
images_test_uneven = h5f_AVA['images_test_uneven']
scores_test_uneven = h5f_AVA['scores_test_uneven']
scores_test_uneven = scores_test_uneven[:,0]

print images_train.shape, scores_train.shape, sample_weights.shape
print images_test_even.shape, scores_test_even.shape
print images_test_uneven.shape, scores_test_uneven.shape

## preprocess function for each batch
def preprocess_input(x, dim_ordering='default'):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        dim_ordering: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    x = np.transpose(x,(0,2,3,1))
    x = x.astype(np.float32,copy=False)
    if dim_ordering == 'th':
        # 'RGB'->'BGR'
#         x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] -= 103.939 # blue
        x[:, 1, :, :] -= 116.779 # green
        x[:, 2, :, :] -= 123.68  # red
    else:
        # 'RGB'->'BGR'
#         x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] -= 103.939 # blue
        x[:, :, :, 1] -= 116.779 # green
        x[:, :, :, 2] -= 123.68  # red
    return x

## define the model and train the model
ff = open('/data/bjin/MyAesthetics/logs/AVA_Resnet_score.txt','w+')
# with K.tf.device('/gpu:0'):
    ## load the network
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet')
x = base_model.output
x = Flatten()(x)
x = Dense(1,init='glorot_uniform')(x)

# this is the model we will train
model = Model(input=input_tensor, output=x)
myadam = Adam(lr=0.001)

# first finetune the top layer
print >>ff, 'finetune the top layer'
for l in model.layers[:-1]:
    l.trainable = False
model.compile(optimizer=myadam, loss='mse')

checkpointer = ModelCheckpoint(filepath="/data/bjin/MyAesthetics/model_weights/AVA_Resnet50_score.hdf5", 
                                verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=2)
mytensorboard = TensorBoard(log_dir='/data/bjin/MyAesthetics/logs/Resnet', histogram_freq=2, write_graph=True)

history = model.fit_generator(loop_batch(images_train,scores_train,sample_weights,batch_size=64,pre_f=preprocess_input),
        samples_per_epoch=len(scores_train), nb_epoch=10, callbacks=[checkpointer, mytensorboard], verbose=1,
        validation_data=loop_batch(images_test_even,scores_test_even,batch_size=64,pre_f=preprocess_input),
        nb_val_samples=len(scores_test_even), max_q_size=5, nb_worker=1, pickle_safe=False, initial_epoch=0)

rint >>ff, history.history

# then finetune all the layers
print >>ff, 'finetune all the layers'
for l in model.layers[:-1]:
    l.trainable = True
model.compile(optimizer=myadam, loss='mse')

checkpointer = ModelCheckpoint(filepath="/data/bjin/MyAesthetics/model_weights/AVA_Resnet50_score.hdf5", 
                                verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=2)
mytensorboard = TensorBoard(log_dir='/data/bjin/MyAesthetics/logs/Resnet', histogram_freq=2, write_graph=True)

history = model.fit_generator(loop_batch(images_train,scores_train,sample_weights,batch_size=64,pre_f=preprocess_input),
        samples_per_epoch=len(scores_train), nb_epoch=10, callbacks=[checkpointer, mytensorboard], verbose=1,
        validation_data=loop_batch(images_test_even,scores_test_even,batch_size=64,pre_f=preprocess_input),
        nb_val_samples=len(scores_test_even), max_q_size=5, nb_worker=1, pickle_safe=False, initial_epoch=0)

print >>ff, history.history
