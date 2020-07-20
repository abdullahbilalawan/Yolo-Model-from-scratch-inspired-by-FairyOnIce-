'''THIS CODE REPRESENTS KERAS IMPLEMENTATION OF TINY YOLO '''

'''---------------------------LIBRARIES USED---------------------------------'''
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import random
import os
from backend import *







'''=============================STEP 1 CONSTANTS DECLARATION==========================='''

NORM_H, NORM_W = 416, 416
GRID_H, GRID_W = 13 , 13
BATCH_SIZE = 8
BOX = 5
ORIG_CLASS = 20


LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

THRESHOLD = 0.2
ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0



'''=============================================STEP 2 DECLARING DIRECTORIES============================'''

train_image_folder = r'JPEGImages\\'
train_annot_folder = r'Annotations\\'

'''=======================================STEP 3 Preprocessing util functions==========================================='''

# normalize the image function
def normalize(image):
    return image / 255.

# loading dataset function
def loading_datset():

    np.random.seed(10)
    train_image, seen_train_labels = parse_annotation(train_annot_folder,
                                                      train_image_folder,
                                                      labels=LABELS)
    print("N train = {}".format(len(train_image)))
    return train_image,seen_train_labels


#check?
# train_image, train_labels= loading_datset()
# print(train_image)


'''=======================================================Batch generator======================================================'''


def generate_batch():
    _ANCHORS01 = np.array([0.08285376, 0.13705531,
                           0.20850361, 0.39420716,
                           0.80552421, 0.77665105,
                           0.42194719, 0.62385487])
    GRID_H, GRID_W = 13, 13
    ANCHORS = _ANCHORS01
    ANCHORS[::2] = ANCHORS[::2] * GRID_W
    ANCHORS[1::2] = ANCHORS[1::2] * GRID_H

    IMAGE_H, IMAGE_W = 416, 416
    BATCH_SIZE = 10
    TRUE_BOX_BUFFER = 50
    generator_config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'LABELS': LABELS,
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER,
    }
    train_image, seen_train_labels = parse_annotation(train_annot_folder,
                                                      train_image_folder,
                                                      labels=LABELS)

    train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                                 norm=normalize, shuffle=True)
    return train_batch_generator
# test
# generate_batch()
