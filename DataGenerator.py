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

import segmentation_models as sm
def load_dataset_segmentation (img_h, img_w, preprocess_type, batch_size=16):
    train2 = util.restructure_data_frame('Severstal_Dataset\\train.csv')
    idx = int(0.8*len(train2)); print()
    preprocess = sm.get_preprocessing(preprocess_type)

    train_batches =  SegmentationDataGenerator(train2.iloc[:idx], img_h=img_h, img_w = img_w, shuffle=True, preprocess=preprocess, batch_size=batch_size,
                                    width_shift_range=20, height_shift_range=20, zoom_range=0.4, 
                                    flip_h=True, flip_v=True, brightness=0.4)

    valid_batches = SegmentationDataGenerator(train2.iloc[idx:],img_h=img_h, img_w = img_w, shuffle=True, preprocess=preprocess, batch_size=batch_size)
    
    """
    iterator = iter(train_batches)
    for _ in range(20):
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


def load_dataset_classification (img_h, img_w, preprocess_type, batch_size=16):
    train2 = util.restructure_data_frame('Severstal_Dataset\\train.csv')
    idx = int(0.8*len(train2)); print()
    preprocess = sm.get_preprocessing(preprocess_type)

    train_batches =  ClassificationDataGenerator(train2.iloc[:idx], img_h=img_h, img_w = img_w, shuffle=True, preprocess=preprocess, batch_size=batch_size,
                                    width_shift_range=20, height_shift_range=20, zoom_range=0.4, 
                                    flip_h=True, flip_v=True, brightness=0.4)

    valid_batches = ClassificationDataGenerator(train2.iloc[idx:],img_h=img_h, img_w = img_w, shuffle=True, preprocess=preprocess, batch_size=batch_size)
    
    """
    iterator = iter(train_batches)
    for _ in range(20):
        images, defect = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)

            print (defect[i])
            util.show_imgs((image,image),('orig','orig'),('',''))
    """

    return train_batches, valid_batches 









from tqdm import tqdm
def test_model(model, img_h, img_w, preprocess_type):
    train2 = util.restructure_data_frame('Severstal_Dataset\\test.csv')
    preprocess = sm.get_preprocessing(preprocess_type)
    bs = 2
    
    test_batches = SegmentationDataGenerator(train2, img_h=img_h, img_w=img_w, preprocess=preprocess, batch_size=bs, subset='test')

    n_samples = 500
    dice_res = 0
    iterator = iter(test_batches)
    for _ in tqdm(range(n_samples)):
        images, masks = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]

            
            mask = mask * 255
            util.show_imgs((image,
                            mask[:,:,0], mask[:,:,1],
                            mask[:,:,2], mask[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))
            


            res = model.predict(np.reshape(image,(1,img_h,img_w,3)))
            res = np.reshape(res,(img_h,img_w,4))
            res[np.where(res < 0.5)] = 0
            res[np.where(res >= 0.5)] = 1

            dice_res += dice_coef(mask.astype(np.uint8), res.astype(np.uint8))
            
            
            res = res * 255
            util.show_imgs((image,
                            res[:,:,0], res[:,:,1],
                            res[:,:,2], res[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))
            
    print (dice_res/(n_samples*bs))


from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    ret = (2 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return ret









"""
def show_examples(train2):
    # DEFECTIVE IMAGE SAMPLES
    filenames = {}
    defects = list(train2[train2['e1']!=''].sample(3).index)
    defects += list(train2[train2['e2']!=''].sample(3).index)
    defects += list(train2[train2['e3']!=''].sample(7).index)
    defects += list(train2[train2['e4']!=''].sample(3).index)

    # DATA GENERATOR
    train_batches = DataGenerator(train2[train2.index.isin(defects)], shuffle=True, info=filenames)
    print('Images and masks from our Data Generator')
    print('KEY: yellow=defect1, green=defect2, blue=defect3, magenta=defect4')

    # DISPLAY IMAGES WITH DEFECTS
    for i,batch in enumerate(train_batches):
        plt.figure(figsize=(14,50)) #20,18
        for k in range(8):
            plt.subplot(8,1,k+1)
            img = batch[0][k,]
            img = Image.fromarray(img.astype('uint8'))
            img = np.array(img)
            extra = '  has defect'
            for j in range(4):
                msk = batch[1][k,:,:,j]
                msk = mask2pad(msk,pad=3)
                msk = mask2contour(msk,width=2)
                if np.sum(msk)!=0: extra += ' '+str(j+1)
                if j==0: # yellow
                    img[msk==1,0] = 235 
                    img[msk==1,1] = 235
                elif j==1: img[msk==1,1] = 210 # green
                elif j==2: img[msk==1,2] = 255 # blue
                elif j==3: # magenta
                    img[msk==1,0] = 255
                    img[msk==1,2] = 255
            plt.title(filenames[16*i+k]+extra)
            plt.axis('off') 
            plt.imshow(img)
        plt.subplots_adjust(wspace=0.05)
        plt.show()
"""
"""
train_df = pd.read_csv(training_dir + '\\train.csv')
data = np.array(train_df)
new_data = []

for i in range(len(data)):
    for j in range(4):
        num = j+1
        if num == data[i][1]:
            new_data.append([data[i][0]+"_"+str(num), data[i][2]])
        else:
            new_data.append([data[i][0]+"_"+str(num)])

original_train_csv = pd.DataFrame(new_data, columns=["ImageId_ClassId","EncodedPixels"])
original_train_csv

original_train_csv.to_csv("original_train.csv", index=None)
"""