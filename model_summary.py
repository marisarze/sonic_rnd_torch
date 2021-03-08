import os
import sys
import json
import random
import math
import numpy as np
from decimal import Decimal
import random
import time
import cv2
from collections import deque
from matplotlib import pyplot as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import queue
from multiprocessing import Queue, Process, Lock

def create_reward_net(state_len, fast=True):
    from keras.models import load_model
    from keras.optimizers import Adam
    from keras.optimizers import SGD
    from keras.models import load_model
    from keras.models import Sequential
    from keras.layers import Input, Dense
    from keras.models import clone_model
    from keras.layers import BatchNormalization
    from keras.layers import TimeDistributed
    from keras.layers import CuDNNLSTM
    from keras.layers import Conv2D
    from keras.layers import Conv2DTranspose
    from keras.layers import Concatenate
    from keras.layers import Reshape
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Reshape
    from keras.layers import UpSampling2D
    from keras.layers import add
    from keras.layers import Activation
    from keras.layers import Lambda
    from keras.layers import average
    from keras.layers import PReLU
    from keras.layers import LeakyReLU
    from keras import losses
    from keras import regularizers
    from keras.models import Model
    from keras import backend as K
    from keras.losses import mean_squared_error
    from keras.losses import categorical_crossentropy
    height = 84
    width = 120
    activ = LeakyReLU(alpha=0.3)
    
    def reward_loss():
        def custom_loss(y_true, y_pred):
            return y_pred
        return custom_loss

    def last_image(tensor):
        return tensor[:,-1,:]
    def ireward(tensor):
        return 0.5 * K.mean(K.pow(tensor[0] - tensor[1], 2), axis=-1)

    state_input = Input(shape=(state_len, height, width, 3))
    float_input = K.cast(state_input, dtype='float32')
    float_input = Lambda(lambda input: input/255.0-0.5)(float_input)
    new_input = Lambda(last_image)(float_input)
    xs = Conv2D(32, (4,4), activation=activ, strides=(2,2), padding='same')(new_input)
    xs = Conv2D(64, (4,4), activation=activ, strides=(2,2), padding='same')(xs)
    xs = Conv2D(64, (4,4), activation=activ, strides=(2,2), padding='same')(xs)
    xs = Conv2D(128, (4,4), activation=activ, strides=(2,2), padding='same')(xs)
    #xs = Conv2D(128, (4,4), activation=activ, strides=(2,2), padding='same')(xs)
    stochastic_output = Flatten()(xs)
    stochastic_part = Model(inputs=state_input, outputs=stochastic_output)
    for layer in stochastic_part.layers:
        layer.trainable = False

    xt = Conv2D(32, (4,4), activation=activ, strides=(2,2), padding='same')(new_input)
    xt = Conv2D(64, (4,4), activation=activ, strides=(2,2), padding='same')(xt)
    xt = Conv2D(64, (4,4), activation=activ, strides=(2,2), padding='same')(xt)
    xt = Conv2D(128, (4,4), activation=activ, strides=(2,2), padding='same')(xt)
    #xt = Conv2D(128, (4,4), activation=activ, strides=(2,2), padding='same')(xt)
    target_output = Flatten()(xt)


    intrinsic_reward = Lambda(ireward)([stochastic_output, target_output])
    model = Model(inputs=state_input, outputs=intrinsic_reward)
    if fast:
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
        #adam = SGD(lr= 1e-4, momentum=0.9)
    else:
        adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-5, decay=0.0, amsgrad=False)
        #adam = SGD(lr= 1e-6, momentum=0.9)
    model.compile(optimizer=adam, loss=reward_loss())
    model.summary()
    return model

A = create_reward_net(1)