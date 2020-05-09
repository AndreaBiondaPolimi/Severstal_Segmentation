import keras
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import utils as util
from keras.preprocessing.image import apply_affine_transform
import pandas as pd
import cv2
import tensorflow as tf

path = 'Severstal_Dataset'
train_path = 'Severstal_Dataset\\train_images\\images_all\\imgs\\'
test_path = 'Severstal_Dataset\\test_images\\'

class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, df, shapes=((4,256,1600),), subset="train", shuffle=False, 
                 preprocess=None, info={}, augmentation_parameters = None, fill_mode='constant'):
        super().__init__()
        self.df = df
        self.shapes = shapes
        self.shape_idx = 0

        self.shuffle = shuffle
        self.subset = subset
        self.preprocess = preprocess
        self.info = info

        self.augmentation_parameters = augmentation_parameters
        self.fill_mode = fill_mode

        if self.subset == "train":
            self.data_path = train_path
        elif self.subset == "test":
            self.data_path = test_path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    

    def on_epoch_end(self):
        self.batch_size = self.shapes[self.shape_idx][0]
        self.img_h = self.shapes[self.shape_idx][1]
        self.img_w = self.shapes[self.shape_idx][2]
        
        #At each epoch change the random crop shape
        self.shape_idx += 1
        if (self.shape_idx >= len(self.shapes)):
            self.shape_idx = 0

        #At each epoch shuffle indexes, if shuffle is true
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __getitem__(self, index): 
        X = np.empty((self.batch_size,self.img_h,self.img_w,3),dtype=np.float32)
        y = np.empty((self.batch_size,self.img_h,self.img_w,4),dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #Collect and crop images of the current batch
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            img = np.asarray(Image.open(self.data_path + f))
            #Generate random crop indexes
            random_crop_indexes = util.get_random_crop_indexes((256,1600),(self.img_h,self.img_w), img)

            for j in range(4):
                #Generate mask
                mask = util.rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])  
                #Random Crop              
                X[i,], y[i,:,:,j] = util.random_crop(img, mask, random_crop_indexes)


        #Data augmentation
        if (self.augmentation_parameters is not None):
            for i in range(len(X)):
                affine_aug, color_aug = util.get_augmentation (self.augmentation_parameters)
                X[i], y[i] = util.augment(affine_aug, X[i].astype(np.uint8), y[i].astype(np.uint8))
                X[i] = util.augment(color_aug, X[i].astype(np.uint8))
            
        #Apply data preprocessing according to the model chosen
        if self.preprocess!=None: 
            X = self.preprocess(X)  
        
        return X, y








    

