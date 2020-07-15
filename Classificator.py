import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def get_classification_model (input_size = (None, None, 3), pretrained_weights=None, preprocess_type = 'none'):
    if (preprocess_type == 'resnet34'):
        return resent34_class_model(input_size, pretrained_weights)
    elif (preprocess_type == 'resnet50'):
        return resnet50_class_model(input_size, pretrained_weights)
        

from segmentation_models import Unet
from classification_models.keras import Classifiers
def resent34_class_model (input_size, pretrained_weights):
    ResNet34, _ = Classifiers.get('resnet34')

    model = ResNet34(input_shape=input_size, weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    
    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


from segmentation_models import Unet
from classification_models.keras import Classifiers
def resnet50_class_model (input_size, pretrained_weights):
    ResNet50, _ = Classifiers.get('resnet50')

    model = ResNet50(input_shape=input_size, weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model





def train (model, train_dataset, valid_dataset, epochs):
    callbacks = []

    checkpoint_dice = ModelCheckpoint(filepath='check_acc{epoch:02d}.h5', monitor='val_acc',mode='max', period=1, save_best_only=True) 
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

