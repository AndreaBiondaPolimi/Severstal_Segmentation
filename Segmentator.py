import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras
import utils as util

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import categorical_crossentropy

def get_segmentation_model (input_size = (None, None, 3), pretrained_weights=None, preprocess_type = 'none', activation='sigmoid'):
    if (preprocess_type == 'resnet34'):
        return resent34_seg_model(input_size, pretrained_weights, activation)
    elif (preprocess_type == 'efficientnetb3'):
        return efficientnetb3_seg_model(input_size, pretrained_weights, activation)
        

from segmentation_models import Unet
from segmentation_models import FPN
from segmentation_models.losses import bce_jaccard_loss
import runai.ga
def resent34_seg_model (input_size, pretrained_weights, activation):
    classes = 4 if activation == 'sigmoid' else 5
    
    model = Unet('resnet34', encoder_weights='imagenet', input_shape=input_size, classes=classes, activation=activation)
    
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


def efficientnetb3_seg_model (input_size, pretrained_weights, activation):
    classes = 4 if activation == 'sigmoid' else 5
    
    model = FPN('efficientnetb3', encoder_weights='imagenet', input_shape=input_size, classes=classes, activation=activation)

    adam = keras.optimizers.Adam(lr=1e-5)
    
    model.compile(optimizer = adam, loss = bce_dice_loss , metrics=[dice_coef])
    #model.compile(optimizer = adam, loss = bce_jaccard_loss , metrics=[dice_coef])
    #model.compile(optimizer = adam, loss = categorical_crossentropy , metrics=[dice_coef])

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


from tqdm import tqdm
import segmentation_models as sm
def search_segmentation_treshold (valid_batches, seg_model, activation, preprocess_type):
    n_samples = valid_batches.__len__()
    classes = 4 if activation == 'sigmoid' else 5
    
    defects = [0, 1, 2 ,3]
    tresholds = [0.3, 0.4, 0.5 ,0.6, 0.7]

    for defect in defects:
        for treshold in tresholds:           
            dice_res = 0
            iterator = iter(valid_batches)
            for _ in tqdm(range(n_samples)):
                images, masks = next(iterator)

                for i in range(len(images)):
                    image = images[i].astype(np.int16)
                    mask = masks[i]                 

                    seg_preprocess = sm.get_preprocessing(preprocess_type)
                    seg_x = seg_preprocess(image)   

                    res = seg_model.predict(np.reshape(seg_x,(1,256,1600,3)))
                    res = np.reshape(res,(256,1600,classes))
                    tot  = np.zeros_like (res)

                    mask_defect = res[:,:,defect]
                    mask_defect[np.where(mask_defect < treshold)] = 0
                    mask_defect[np.where(mask_defect >= treshold)] = 1

                    tot [:,:,defect] = mask_defect

                    res[np.where(res < 0.5)] = 0
                    res[np.where(res >= 0.5)] = 1


                    #Update dice value
                    dice_res += dice_coef_test(mask.astype(np.uint8), tot.astype(np.uint8))
                    
                    coef = dice_coef_test(mask.astype(np.uint8), tot.astype(np.uint8))
                    util.show_img_and_def((image, mask, tot), ('orig','mask','pred ' + str(coef)))

            print()
            print('Search for defect', defect, 'with treshold', treshold, 'dice', dice_res)





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




def dice_coef_test(y_true, y_pred, smooth=1):  
    y_true_f = y_true[:,:,:4].flatten()
    y_pred_f = y_pred[:,:,:4].flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    ret = (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return ret

