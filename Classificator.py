import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def get_classification_model (input_size = (None, None, 3), pretrained_weights=None, preprocess_type = 'none'):
    return class_model(preprocess_type, input_size, pretrained_weights)
        

from classification_models.keras import Classifiers
def class_model (preprocess_type, input_size, pretrained_weights):
    ResNet, _ = Classifiers.get(preprocess_type)

    model = ResNet(input_shape=input_size, weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    
    adam = keras.optimizers.Adam(lr=1e-3)
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

