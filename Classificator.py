import numpy as np 
import os
import numpy as np
import tensorflow as tf
import keras

from keras import backend as K
from keras.callbacks import ModelCheckpoint

def get_classification_model (input_size = (128,800,3), pretrained_weights=None, preprocess_type = 'none'):
    if (preprocess_type == 'resnet34'):
        return resent34_class_model(input_size, pretrained_weights)
        

from segmentation_models import Unet
from classification_models.keras import Classifiers
def resent34_class_model (input_size, pretrained_weights):
    ResNet34, _ = Classifiers.get('resnet34')

    model = ResNet34(input_shape=input_size, weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(1, activation='softmax')(x)
    model = keras.models.Model(inputs=[model.input], outputs=[output])
    
    #adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

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
          steps_per_epoch=800,
          validation_data=valid_dataset,
          validation_steps=200,
          callbacks=callbacks)
    

    #model.fit_generator(train_dataset, validation_data = valid_dataset, 
                        #epochs = 50, verbose=1,
                        #allbacks=callbacks)

    model.save('seg_final.h5')

