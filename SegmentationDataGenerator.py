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
                 preprocess=None, info={},
                 rotation_range=0, width_shift_range=0, height_shift_range=0,
                 flip_h=False, flip_v=False, fill_mode='constant'):
        super().__init__()
        self.df = df
        self.shapes = shapes
        self.shape_idx = 0

        self.shuffle = shuffle
        self.subset = subset
        self.preprocess = preprocess
        self.info = info

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.flip_h = flip_h
        self.flip_v = flip_v
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

        #print ((self.img_h,self.img_h))
        
        self.shape_idx += 1
        if (self.shape_idx >= len(self.shapes)):
            self.shape_idx = 0

        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __getitem__(self, index): #index restart automatically seems when max is reached
        X = np.empty((self.batch_size,self.img_h,self.img_w,3),dtype=np.float32)
        y = np.empty((self.batch_size,self.img_h,self.img_w,4),dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            
            img = np.asarray(Image.open(self.data_path + f))

            random_crop_indexes = get_random_crop_indexes_v2((256,1600),(self.img_h,self.img_w), img)

            for j in range(4):
                mask = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])  
                #Random Crop              
                X[i,], y[i,:,:,j] = random_crop(img, mask, random_crop_indexes)

        #Data augmentation
        datagen = ImageDataGenerator()
        for i in range(len(X)):
            theta = np.random.uniform(-self.rotation_range,self.rotation_range)
            tx = np.random.uniform(-self.width_shift_range,self.width_shift_range)
            ty = np.random.uniform(-self.height_shift_range,self.height_shift_range)
            f_h = np.random.choice([True, False], p=[0.5, 0.5]) if (self.flip_h) else False
            f_v = np.random.choice([True, False], p=[0.5, 0.5]) if (self.flip_v) else False
            fill_mode = self.fill_mode
        
            X[i] = apply_affine_transform(X[i], theta=theta, tx=tx, ty=ty, fill_mode=fill_mode)
            y[i] = apply_affine_transform(y[i], theta=theta, tx=tx, ty=ty, fill_mode=fill_mode)

            X[i] = datagen.apply_transform(x=X[i], 
                            transform_parameters={'flip_horizontal':f_h, 'flip_vertical':f_v})

            y[i] = datagen.apply_transform(x=y[i], 
                            transform_parameters={'flip_horizontal':f_h, 'flip_vertical':f_v})     

        if self.preprocess!=None: 
            X = self.preprocess(X)  
        
        return X, y



def rle2maskResize(rle):
    height= 256
    width = 1600
    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return np.zeros((height,width) ,dtype=np.uint8)

    mask= np.zeros( width*height ,dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1

    return mask.reshape( (height,width), order='F' )

def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3) 

def mask2pad(mask, pad=2):
    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT
    w = mask.shape[1]
    h = mask.shape[0]
    
    # MASK UP
    for k in range(1,pad,2):
        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK DOWN
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK LEFT
    for k in range(1,pad,2):
        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)
        mask = np.logical_or(mask,temp)
    # MASK RIGHT
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)
        mask = np.logical_or(mask,temp)
    
    return mask 


def get_random_crop_indexes(original_image_size, random_crop_size):
    height, width = original_image_size
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return ((dx, dy), (x,y))

def random_crop(img, mask, random_crop_indexes):
    (dx, dy), (x,y) = random_crop_indexes

    img_cropped = img[y:(y+dy), x:(x+dx), :]
    mask_cropped = mask[y:(y+dy), x:(x+dx)]
    return (img_cropped, mask_cropped)


#Try to get the random crop that does not show full black image, 
#usually steel is on the right or on the left when background is present
def get_random_crop_indexes_v2(original_image_size, random_crop_size, img):
    n_tries_before_default = 1 #Try n times to get the random crop before rx/lx choice
    height, width = original_image_size
    dy, dx = random_crop_size

    for i in range (n_tries_before_default):
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)

        if (not is_total_black(img, x, y, dx, dy)):
            return ((dx, dy), (x,y))

    #Try with left crop
    x = 0
    y = np.random.randint(0, height - dy + 1)
    if (not is_total_black(img, x, y, dx, dy)):
        return ((dx, dy), (x,y))

    #Try with right crop
    x = width - dx - 1
    y = np.random.randint(0, height - dy + 1)

    return ((dx, dy), (x,y))
   


def is_total_black(img, x, y, dx, dy):
    cropped_img = img[y:(y+dy), x:(x+dx), :].copy()
    #plt.imshow(cropped_img)
    #plt.show()

    cropped_img[cropped_img < 30] = 0
    if (np.count_nonzero(cropped_img) > 0):
        return False
    return True
    