import keras
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import utils as util
from keras.preprocessing.image import apply_affine_transform

path = 'Severstal_Dataset'
train_path = 'Severstal_Dataset\\train_images\\images_all\\imgs\\'
test_path = 'Severstal_Dataset\\test_images\\'

class ClassificationDataGenerator(keras.utils.Sequence):
    def __init__(self, df, img_h = 128, img_w=1600, batch_size = 16, subset="train", shuffle=False, 
                 preprocess=None, info={},
                 rotation_range=0, width_shift_range=0, height_shift_range=0,
                 zoom_range=0, flip_h=False, flip_v=False, brightness=0, fill_mode='constant'):
        super().__init__()
        self.df = df
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.brightness = brightness
        self.fill_mode = fill_mode

        
        if self.subset == "train":
            self.data_path = train_path
        elif self.subset == "test":
            self.data_path = test_path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index): 
        X = np.empty((self.batch_size,self.img_h,self.img_w,3),dtype=np.float32)
        y = np.zeros((self.batch_size,1),dtype=np.float32)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(self.data_path + f).resize((self.img_w,self.img_h)) 
            for j in range(4):
                y[i][0] += rle2class(self.df['e'+str(j+1)].iloc[indexes[i]])
            if (y[i][0] > 0):
                y[i][0] = 1
            
        if self.preprocess!=None: 
            X = self.preprocess(X)

        datagen = ImageDataGenerator()
        for i in range(len(X)):
            theta = np.random.uniform(-self.rotation_range,self.rotation_range)
            tx = np.random.uniform(-self.width_shift_range,self.width_shift_range)
            ty = np.random.uniform(-self.height_shift_range,self.height_shift_range)
            z = np.random.uniform(1-self.zoom_range,1+self.zoom_range)
            br = np.random.uniform(1-self.brightness,1+self.brightness)
            f_h = np.random.choice([True, False], p=[0.5, 0.5]) if (self.flip_h) else False
            f_v = np.random.choice([True, False], p=[0.5, 0.5]) if (self.flip_v) else False
            fill_mode = self.fill_mode

            X[i] = apply_affine_transform(X[i], theta=theta, tx=tx, ty=ty, zx=z, zy=z,
                                         fill_mode=fill_mode)

            X[i] = datagen.apply_transform(x=X[i], 
                            transform_parameters={'brightness':br, 'flip_horizontal':f_h, 'flip_vertical':f_v})
        
        
        return X, y


def rle2class(rle):
    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return 0
    
    return 1