import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras
import utils as util

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import categorical_crossentropy

def get_segmentation_model (preprocess_type, input_size = (None, None, 3),  pretrained_weights=None, activation='sigmoid', loss='dice_bce'):
    if (loss == 'dice_bce'):
        loss_type = bce_dice_loss
    elif (loss == 'categorical_crossentropy'):
        loss_type = keras.losses.categorical_crossentropy
    elif (loss == 'tversky'):
        loss_type = tversky_loss
    else:
        raise "Loss type error"

    return seg_model(preprocess_type, input_size, pretrained_weights, activation, loss_type)
        

from segmentation_models import Unet
def seg_model (preprocess_type, input_size, pretrained_weights, activation, loss):
    classes = 4 if activation == 'sigmoid' else 5
    
    model = Unet(preprocess_type, encoder_weights='imagenet', input_shape=input_size, classes=classes, activation=activation)

    adam = keras.optimizers.Adam(lr=1e-4)

    model.compile(optimizer = adam, loss = loss , metrics=[dice_coef])

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
    y_true_f = K.flatten(y_true[:,:,:,:4])
    y_pred_f = K.flatten(y_pred[:,:,:,:4])
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




def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.2
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)