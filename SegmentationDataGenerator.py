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
from random import shuffle

path = 'Severstal_Dataset'
train_path = 'Severstal_Dataset\\train_images\\images_all\\imgs\\'
test_path = 'Severstal_Dataset\\test_images\\'

class SegmentationDataGenerator(keras.utils.Sequence):
    def __init__(self, df, shapes=((1,256,1600),), subset="train", shuffle=False, use_balanced_dataset=False,
                 preprocess=None, info={}, augmentation_parameters = None, use_defective_only=False,
                 activation='sigmoid'):
        
        super().__init__()
        self.df = df
    
        self.shapes = shapes
        self.shape_idx = 0

        self.shuffle = shuffle
        self.subset = subset
        self.preprocess = preprocess
        self.info = info
        self.use_balanced_dataset = use_balanced_dataset

        self.augmentation_parameters = augmentation_parameters

        if self.subset == "train":
            self.data_path = train_path
        elif self.subset == "test":
            self.data_path = test_path

        if activation == 'sigmoid':
            self.channels_mask = 4
        else:
            self.channels_mask = 5

        if self.use_balanced_dataset:
            self.df_classes = self.split_class_from_dataframe(use_defective_only)

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

        #At each epoch shuffle indexes, if shuffle is true and if not use balanced class
        if (not self.use_balanced_dataset):
            self.indexes = np.arange(len(self.df))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
    

    def __getitem__(self, index): 
        X = np.empty((self.batch_size,self.img_h,self.img_w,3),dtype=np.float32)
        y = np.empty((self.batch_size,self.img_h,self.img_w,self.channels_mask),dtype=np.int8)
        mask = np.empty((256,1600,self.channels_mask),dtype=np.int8)
        
        if (self.use_balanced_dataset):
            df_batch = self.get_class_balanced_batch(self.batch_size)
        else:
            df_batch = self.get_standard_batch(index, self.batch_size)

        #Generate random crop indexes, create full resoultion mask and then crop
        for i in range (len(df_batch)):
            df = df_batch[i]
            img = np.asarray(Image.open(self.data_path + df['ImageId']))
            for j in range(4):
                mask[:,:,j] = util.rle2maskResize(df['e'+str(j+1)]) 
            if (self.channels_mask > 4): 
                mask[:,:,4] = util.mask2Background(mask)     

            random_crop_indexes = util.get_random_crop_indexes((256,1600), (self.img_h,self.img_w), img, mask[:,:,:4])
            X[i,], y[i,:,:,:] = util.random_crop(img, mask, random_crop_indexes)
     
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



    def split_class_from_dataframe(self, use_defective_only):
        df_1 = self.df[self.df.e1.astype(bool)]
        df_2 = self.df[self.df.e2.astype(bool)]
        df_3 = self.df[self.df.e3.astype(bool)]
        df_4 = self.df[self.df.e4.astype(bool)]

        if (not use_defective_only):
            df_0 = self.df[self.df['count'] == 0]
            return df_0, df_1, df_2, df_3, df_4 
        
        return df_1, df_2, df_3, df_4 


    def get_class_balanced_batch(self, batch_size):
        assert (batch_size % len(self.df_classes) == 0), "Batch size should be divisible by the number of classes for the moment"

        df_batch = list()
        n_img_per_class = int(batch_size/len(self.df_classes))
        for df in self.df_classes:
            indexes = np.random.choice(len(df), n_img_per_class, replace=True)
            for index in indexes:
                df_batch.append(df.iloc[index])
        shuffle(df_batch)

        return df_batch


    def get_standard_batch (self, index, batch_size):
        indexes = self.indexes[index*batch_size:(index+1)*batch_size]
        df_batch = list()
        for index in indexes:
            df_batch.append(self.df.iloc[index])

        return df_batch
