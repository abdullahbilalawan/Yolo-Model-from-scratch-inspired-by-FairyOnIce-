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
import tensorflow as tf

from backend import *

def tiny_yolo_model():
    model = Sequential()

    #--------------------------Layer1-----------------------------------------

    model.add(Conv2D(16,(3,3),strides=(1,1),padding='same',input_shape=(416,416,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    #-------------------------Layer2--------------------------------------------
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

    # -------------------------Layer3--------------------------------------------
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    # -------------------------Layer4--------------------------------------------
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

    # -------------------------Layer5--------------------------------------------
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

    # -------------------------Layer6--------------------------------------------
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    # model.add(MaxPooling2D(pool_size=(2,2)))

    #
    #
    # -------------------------Layer7--------------------------------------------
    model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    #
    #
    #
    # -------------------------Layer8--------------------------------------------
    model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    #
    # -------------------------Layer9--------------------------------------------
    ANCHORS = np.array([1.07709888, 1.78171903,  # anchor box 1, width , height
                        2.71054693, 5.12469308,  # anchor box 2, width,  height
                        10.47181473, 10.09646365,  # anchor box 3, width,  height
                        5.48531347, 8.11011331])  # anchor box 4, width,  height
    # BOX = int(len(ANCHORS) / 2)
    # model.add(Conv2D (1,(1,1), strides=(1,1), padding='same'))
    model.add(Conv2D(4 * (4 + 1 + 20), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Reshape((13, 13, 4,25)))


    return model

    ###############################################################################



#=================================================Model Summary=================================================

def show_model_structure():
    model = tiny_yolo_model()
    print(model.summary())



#========================CUSTOM LOSS FN===============================================================

GRID_W = 13
GRID_H = 13
BATCH_SIZE = 34
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT = 5.0
LAMBDA_COORD = 1.0
LAMBDA_CLASS = 1.0



ANCHORS = np.array([1.07709888, 1.78171903,  # anchor box 1, width , height
                        2.71054693, 5.12469308,  # anchor box 2, width,  height
                        10.47181473, 10.09646365,  # anchor box 3, width,  height
                        5.48531347, 8.11011331])  # anchor box 4, width,  height

BATCH_SIZE        = 200
IMAGE_H, IMAGE_W  = 416, 416
GRID_H,  GRID_W   = 13 , 13
TRUE_BOX_BUFFER   = 50
BOX               = int(len(ANCHORS)/2)



model, true_boxes = define_YOLOv2(416,416,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,20,
                                  trainable=False)

print(model.summary())

def custom_loss(y_true, y_pred):
    return (custom_loss_core(
        y_true,
        y_pred,
        true_boxes,
        GRID_W,
        GRID_H,
        BATCH_SIZE,
        ANCHORS,
        LAMBDA_COORD,
        LAMBDA_CLASS,
        LAMBDA_NO_OBJECT,
        LAMBDA_OBJECT))