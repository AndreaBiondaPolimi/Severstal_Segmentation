import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def get_segmentation_model (input_size = (None, None, 3), pretrained_weights=None, preprocess_type = 'none'):
    if (preprocess_type == 'resnet34'):
        return resent34_seg_model(input_size, pretrained_weights)
    elif (preprocess_type == 'efficientnetb3'):
        return efficientnetb3_seg_model(input_size, pretrained_weights)
        

from segmentation_models import Unet
from segmentation_models import FPN
from segmentation_models.losses import bce_jaccard_loss
import runai.ga
def resent34_seg_model (input_size, pretrained_weights):
    model = Unet('resnet34', encoder_weights='imagenet', input_shape=input_size, classes=4, activation='sigmoid')
    
    #loss = 'binary_crossentropy'
    loss = bce_dice_loss
    #loss = bce_jaccard_loss

    adam = keras.optimizers.Adam(lr=1e-5)
    #adam = runai.ga.keras.optimizers.Optimizer(adam, steps=3)

    model.compile(optimizer = adam, loss = loss , metrics=[dice_coef])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def efficientnetb3_seg_model (input_size, pretrained_weights):
    model = FPN('efficientnetb3', encoder_weights='imagenet', input_shape=input_size, classes=4, activation='sigmoid')

    adam = keras.optimizers.Adam(lr=1e-4)
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer = adam, loss = bce_dice_loss , metrics=[dice_coef])
    #model.compile(optimizer = adam, loss = lovasz_loss , metrics=[dice_coef])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model





def train (model, train_dataset, valid_dataset, epochs):
    callbacks = []

    checkpoint_dice = ModelCheckpoint(filepath='check_val_dice{epoch:02d}.h5', monitor='val_dice_coef',mode='max', period=1, save_best_only=True) 
    scheduler = ReduceLROnPlateau(monitor='val_loss', mode="min", patience=3, factor=0.5, verbose=True)

    callbacks.append(checkpoint_dice)
    callbacks.append(scheduler)
    
    
    
    model.fit(x=train_dataset,
          epochs=epochs,  #### set repeat in training dataset
          steps_per_epoch=train_dataset.__len__(),
          validation_data=valid_dataset,
          validation_steps=valid_dataset.__len__(),
          callbacks=callbacks,
          shuffle=True)
    
    model.save('seg_final.h5')






def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    #return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

