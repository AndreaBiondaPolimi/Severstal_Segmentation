import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2
import utils as util
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import resize_images
from datetime import datetime


def load_dataset_v2 (img_h, img_w, batch_size, preprocess_type='None'):
    training_dir = 'Severstal_Dataset'

    tr = pd.read_csv(training_dir + '/train.csv')
    print(len(tr))
    
    df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
    print(len(df_train))

    print (df_train.head())

    def rle2mask(rle, imgshape):
        width = imgshape[0]
        height= imgshape[1]
        
        mask= np.zeros( width*height ).astype(np.uint8)
        
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]
            
        return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


    
    def keras_generator(batch_size):
        while True:
            x_batch = []
            y_batch = []
            
            for i in range(batch_size):            
                fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]
                img = cv2.imread( training_dir + '\\train_images\\' +fn )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
                
                mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)
                
                img = cv2.resize(img, (1600, 256))
                mask = cv2.resize(mask, (1600, 256))
                
                x_batch += [img]
                y_batch += [mask]
                                        
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            yield x_batch, np.expand_dims(y_batch, -1)


    for x, y in keras_generator(100):
        print(x.shape, y.shape)
        for i in range(len(x)):
            plt.imshow(x[i])
            plt.show()
            plt.imshow(np.reshape(y[i],(256,1600)))
            plt.show()
        
    
    return None

def load_dataset(img_h, img_w, batch_size, preprocess_type='None'):
    # Set the seed for random operations. 
    # This let our experiments to be reproducible. 
    SEED = 1234
    tf.random.set_seed(SEED)  

    # Batch size
    bs = batch_size

    #Load dataset
    train_orig, train_bin, _ = util.load_train_dataset()


    if (preprocess_type=='None'):
        train_img_data_gen = ImageDataGenerator(rotation_range=2,
                                                width_shift_range=2,
                                                height_shift_range=2,
                                                zoom_range=0.3,
                                                #horizontal_flip=True,
                                                #vertical_flip=True,
                                                cval=0,
                                                rescale=1./65535)
    

    train_mask_data_gen = ImageDataGenerator(rotation_range=2,
                                             width_shift_range=2,
                                             height_shift_range=2,
                                             zoom_range=0.3,
                                             #horizontal_flip=True,
                                             #vertical_flip=True,
                                             cval=0,
                                             rescale=1./65535)



    # Training
    # Two different generators for images and masks
    # ATTENTION: here the seed is important!! We have to give the same SEED to both the generator
    # to apply the same transformations/shuffling to images and corresponding masks

    """
    for i in range (len(train_orig)):
        augmented_img = train_orig[i]   # First element

        target_img = train_bin[i]   # First element
        
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(np.reshape(augmented_img,(img_h,img_w)))
        f.add_subplot(1,2, 2)
        plt.imshow(np.reshape(target_img,(img_h,img_w)))
        plt.show(block=True)
    """
    
    ### TRAINING FLOW FROM DIRECTORY ###
    train_img_gen = train_img_data_gen.flow(train_orig, shuffle=True, batch_size=bs, seed=SEED)

    train_mask_gen = train_mask_data_gen.flow(train_bin, shuffle=True, batch_size=bs, seed=SEED)

    #print (len(train_img_gen))

    ### VALIDATION FLOW FROM DIRECOTRY ###
    """
    valid_img_gen = train_img_data_gen.flow_from_directory(os.path.join(dataset_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs, 
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED,
                                                       subset='validation')  
    
    valid_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(dataset_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         color_mode='grayscale',
                                                         shuffle=True,
                                                         interpolation='bilinear',
                                                         seed=SEED,
                                                         subset='validation')
    """
    #print (len(valid_img_gen))

                                         
    train_gen = zip(train_img_gen, train_mask_gen)
    #valid_gen = zip(valid_img_gen, valid_mask_gen)

    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 1], [None, img_h, img_w, 1]))

    """
    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 1], [None, img_h, img_w, 1]))
    """

    """
    def prepare_target(x_, y_):
        y_ = tf.cast(y_, tf.int32)
        return x_, y_
    """

    #train_dataset = train_dataset.map(prepare_target)
    train_dataset = train_dataset.repeat()

    """
    valid_dataset = valid_dataset.map(prepare_target)
    valid_dataset = valid_dataset.repeat()
    """
    
    """
    iterator = iter(train_dataset)
    
    for _ in range(3):
        augmented_imgs, targets = next(iterator)   
        for i in range (len(augmented_imgs)):
            augmented_img = augmented_imgs[i]   # First element
            augmented_img = augmented_img * 65535  # denormalize

            target_img = targets[i]   # First element
            target_img = target_img * 65535  # denormalize

            
            f = plt.figure()
            f.add_subplot(1,2, 1)
            plt.imshow(np.reshape(augmented_img,(img_h,img_w)))
            f.add_subplot(1,2, 2)
            plt.imshow(np.reshape(target_img,(img_h,img_w)))
            plt.show(block=True)
    """
    """
    iterator = iter(valid_dataset)
  
    for _ in range(3):
        augmented_img, target = next(iterator)    
        augmented_img = augmented_img[0]   # First element
        augmented_img = augmented_img * 255  # denormalize
        augmented_img = tf.dtypes.cast(augmented_img, tf.int32)

        target_img = target[0]   # First element
        target_img = target_img * 255  # denormalize

        
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(augmented_img)
        f.add_subplot(1,2, 2)
        plt.imshow(np.reshape(target_img,(img_h,img_w)))
        plt.show(block=True)
    """
    
    return train_dataset #,valid_dataset



def test_model(model, to_show, img_h, img_w, preprocess_type='none'):
    test_orig, test_bin, test_view = util.load_test_dataset()
    #test_orig, test_bin, test_view = util.load_train_dataset()
    model.load_weights('seg_final.h5')

    for i in range(len(test_orig)):
        img = test_orig[i]
        tgt = test_bin[i]
        view = test_view[i]

        if (preprocess_type=='none'):
            img_array = img / 65535
            bin_array = tgt / 65535


        #Image prediction
        img_array = np.reshape(img_array,(1,img_h,img_w,1))
        res = model.predict(img_array)
        
        #res[np.where(res < 0.5)] = 0
        #res[np.where(res >= 0.5)] = 1

        if (to_show == True):
            util.show_test_results(res, bin_array, view)
    


def rle_encode(img):
      # Flatten column-wise
      pixels = img.T.flatten()
      pixels = np.concatenate([[0], pixels, [0]])
      runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
      runs[1::2] -= runs[::2]
      return ' '.join(str(x) for x in runs)
