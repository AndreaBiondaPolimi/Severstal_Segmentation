import keras
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import utils as util
from keras.preprocessing.image import apply_affine_transform
from SegmentationDataGenerator import SegmentationDataGenerator
from ClassificationDataGenerator import ClassificationDataGenerator
import timeit

augmentation_parameters = {'flip_prob':0.5, 'shift_limit':0.1, 'rotate_limit':20, 'shift_rot_prob':0.5, 
                           'contrast_limit':0.2, 'brightness_limit':0.2, 'contr_bright_prob':0.5}

import segmentation_models as sm
def load_dataset_segmentation (preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\train.csv')
    idx = int(0.8*len(train2)); print()

    preprocess = sm.get_preprocessing(preprocess_type)
    shapes = ((10,256,256), (10,256,512), (10,256,608))

    train_batches =  SegmentationDataGenerator(train2.iloc[:idx], shapes=shapes, shuffle=True, use_balanced_dataset=True,
                                                preprocess=preprocess, augmentation_parameters=augmentation_parameters)

    valid_batches = SegmentationDataGenerator(train2.iloc[idx:], shuffle=True, preprocess=preprocess)
    
    """
    iterator = iter(train_batches)
    for _ in range(100):
        images, masks = next(iterator)
        masks = masks * 255

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]

            util.show_imgs((image,
                            mask[:,:,0], mask[:,:,1],
                            mask[:,:,2], mask[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))
    """ 
    
    return train_batches, valid_batches 




def load_dataset_classification (preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\train.csv')
    idx = int(0.8*len(train2)); print()

    preprocess = sm.get_preprocessing(preprocess_type)
    shapes = ((8,256,1600),)

    train_batches =  ClassificationDataGenerator(train2.iloc[:idx], shuffle=True, preprocess=preprocess, shapes=shapes,
                                                    augmentation_parameters=augmentation_parameters)

    valid_batches = ClassificationDataGenerator(train2.iloc[idx:], shuffle=True, preprocess=preprocess)

    """
    iterator = iter(train_batches)
    for _ in range(20):
        images, defect = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.uint8)

            print (defect[i])
            util.show_imgs((image,image),('orig','orig'),('',''))
    """

    return train_batches, valid_batches 









from tqdm import tqdm
def test_model(seg_model, cla_model, preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\test.csv')
    preprocess = sm.get_preprocessing(preprocess_type)
    bs = 2
    
    test_batches = SegmentationDataGenerator(train2, preprocess=preprocess, shapes=((bs,256,1600),), subset='test')

    n_samples = test_batches.__len__()
    dice_res = 0
    iterator = iter(test_batches)
    for _ in tqdm(range(n_samples)):
        images, masks = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]

            
            util.show_imgs((image,
                            mask[:,:,0], mask[:,:,1],
                            mask[:,:,2], mask[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))
            
            
            #Predict if the current image is defective or not
            cls_res = cla_model.predict(np.reshape(image,(1,256,1600,3)))

            #If it is most probable defective, the result is a whole zero mask
            if (cls_res < 0.5):
                res = np.zeros((256,1600,4),dtype=np.int8)

            #Otherwise apply segmentation 
            else:
                res = seg_model.predict(np.reshape(image,(1,256,1600,3)))
                
                res = np.reshape(res,(256,1600,4))
                res[np.where(res < 0.5)] = 0
                res[np.where(res >= 0.5)] = 1

            #Update dice value
            dice_res += dice_coef(mask.astype(np.uint8), res.astype(np.uint8))
            coef = dice_coef(mask.astype(np.uint8), res.astype(np.uint8))
            
            
            util.show_imgs((image,
                            res[:,:,0], res[:,:,1],
                            res[:,:,2], res[:,:,3]),
                            (str(coef),'1','2','3','4'),('','','','',''))
            

    print (dice_res/(n_samples*bs))


from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    ret = (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return ret

