import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, BatchNormalization, Activation, add, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, MaxPooling2D, concatenate, UpSampling2D, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


def get_segmentation_model (input_size = (128,800,3), pretrained_weights=None, preprocess_type = 'none'):
    if (preprocess_type == 'resnet34'):
        return resent34_seg_model(input_size, pretrained_weights)
    elif (preprocess_type == 'efficientnetb4'):
        return efficientnetb4_seg_model(input_size, pretrained_weights)
        

from segmentation_models import Unet
#from segmentation_models.backbones import get_preprocessing
def resent34_seg_model (input_size, pretrained_weights):
    model = Unet('resnet34', encoder_weights='imagenet', input_shape=input_size, classes=4, activation='sigmoid')
    
    #model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer = 'adam', loss = bce_dice_loss , metrics=[dice_coef])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def efficientnetb4_seg_model (input_size, pretrained_weights):
    model = Unet('efficientnetb4', encoder_weights='imagenet', input_shape=input_size, classes=4, activation='sigmoid')


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

    #es_callback = EarlyStopping(monitor='val_loss', patience=70)
    #checkpoint_loss = ModelCheckpoint(filepath='check_val_loss{epoch:02d}.h5', monitor='val_loss',mode='min', period=1, save_best_only=True) 
    checkpoint_dice = ModelCheckpoint(filepath='check_val_dice{epoch:02d}.h5', monitor='val_dice_coef',mode='max', period=1, save_best_only=True) 


    #callbacks.append(es_callback)
    #callbacks.append(checkpoint_loss)
    callbacks.append(checkpoint_dice)

    
    model.fit(x=train_dataset,
          epochs=epochs,  #### set repeat in training dataset
          steps_per_epoch=300,
          validation_data=valid_dataset,
          validation_steps=200,
          callbacks=callbacks)
    

    #model.fit_generator(train_dataset, validation_data = valid_dataset, 
                        #epochs = 50, verbose=1,
                        #allbacks=callbacks)

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






def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), grad, 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels