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
    shapes = ((5,256,256),)

    train_batches =  SegmentationDataGenerator(train2.iloc[:idx], shapes=shapes, shuffle=True, use_balanced_dataset=True,
                                                preprocess=preprocess, augmentation_parameters=augmentation_parameters)

    valid_batches = SegmentationDataGenerator(train2.iloc[idx:], shuffle=True, preprocess=preprocess)
    
    """
    iterator = iter(train_batches)
    for _ in range(100):
        images, masks = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]
            util.show_img_and_def((image, mask, ('orig','mask'))
    """
    
    return train_batches, valid_batches 




def load_dataset_classification (preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\train.csv')
    idx = int(0.8*len(train2)); print()

    preprocess = sm.get_preprocessing(preprocess_type)
    shapes = ((4,256,1600),)

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
def test_model(seg_model, cla_model, seg_preprocess_type, cls_preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\test.csv')
    seg_preprocess = sm.get_preprocessing(seg_preprocess_type)
    cls_preprocess = sm.get_preprocessing(cls_preprocess_type)
    bs = 2
    
    test_batches = SegmentationDataGenerator(train2, shapes=((bs,256,1600),), subset='test')

    n_samples = test_batches.__len__()
    dice_res = 0
    iterator = iter(test_batches)
    for _ in tqdm(range(n_samples)):
        images, masks = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]    

            seg_x = seg_preprocess(image)          
            cls_x = cls_preprocess(image)          
               
            #Predict if the current image is defective or not
            cls_res = cla_model.predict(np.reshape(cls_x,(1,256,1600,3)))

            #If it is most probable defective, the result is a whole zero mask
            
            if (cls_res < 0):
                res = np.zeros((256,1600,4),dtype=np.uint8)

            #Otherwise apply segmentation 
            else:
                res = seg_model.predict(np.reshape(seg_x,(1,256,1600,3)))
                res = np.reshape(res,(256,1600,4))

            res[np.where(res < 0.5)] = 0
            res[np.where(res >= 0.5)] = 1

            #Update dice value
            dice_res += dice_coef(mask.astype(np.uint8), res.astype(np.uint8))
            
            #coef = dice_coef(mask.astype(np.uint8), res.astype(np.uint8))
            #util.show_img_and_def((image, mask, res), ('orig','mask','pred ' + str(coef)))  

    print (dice_res/(n_samples*bs))

acc_matrix = [0, 0, 0, 0]
def fulltest_classification_model (cls_model, cls_preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\test.csv')
    cls_preprocess = sm.get_preprocessing(cls_preprocess_type)
    bs = 2
    
    test_batches = ClassificationDataGenerator(train2, shapes=((bs,256,1600),), subset='test')

    n_samples = test_batches.__len__()
    iterator = iter(test_batches)
    for _ in tqdm(range(n_samples)):
        images, trues = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            true = trues[i]    
         
            cls_x = cls_preprocess(image)          
               
            #Predict if the current image is defective or not
            cls_res = cls_model.predict(np.reshape(cls_x,(1,256,1600,3)))

            update_acc_matrix (true, cls_res)

    print ('True positive: ' , acc_matrix[0], 'False positive: ', acc_matrix[1])
    print ('False negative: ', acc_matrix[2], 'True Negative: ' , acc_matrix[3])



from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    ret = (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return ret


def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return np.mean(np.equal(y_true, np.round(y_pred)))


def update_acc_matrix (y_true, y_pred):
    y_pred = np.round(y_pred)
    if (y_true == y_pred):
        if (y_true == 1):
            acc_matrix[0] += 1
        else:
            acc_matrix[3] += 1
    else:
        if (y_true == 0):
            acc_matrix[1] += 1
        else:
            acc_matrix[2] += 1
