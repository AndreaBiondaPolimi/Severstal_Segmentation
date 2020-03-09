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


def load_dataset(img_h, img_w, batch_size, preprocess_type='None'):
    # Set the seed for random operations. 
    # This let our experiments to be reproducible. 
    SEED = 1234
    tf.random.set_seed(SEED)  

    #Target directory
    dataset_dir = "Severstal_Dataset"

    # Batch size
    bs = batch_size

    train_img_data_gen = ImageDataGenerator(rotation_range=2,
                                            width_shift_range=2,
                                            height_shift_range=2,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='reflect',
                                            cval=0,
                                            validation_split=0.2,
                                            rescale=1./255)


    train_mask_data_gen = ImageDataGenerator(rotation_range=2,
                                             width_shift_range=2,
                                             height_shift_range=2,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='reflect',
                                             cval=0,
                                             validation_split=0.2,
                                             rescale=1./255)



    # Training
    # Two different generators for images and masks
    # ATTENTION: here the seed is important!! We have to give the same SEED to both the generator
    # to apply the same transformations/shuffling to images and corresponding masks
    training_dir = os.path.join(dataset_dir, 'train_images')
    
    ### TRAINING FLOW FROM DIRECTORY ###
    train_img_gen = train_img_data_gen.flow_from_directory(os.path.join(training_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs, 
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED,
                                                       subset='training')  
    
    train_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(training_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         shuffle=True,
                                                         color_mode='rgba',
                                                         interpolation='bilinear',
                                                         seed=SEED,
                                                         subset='training')

    #print (len(train_img_gen))

    ### VALIDATION FLOW FROM DIRECOTRY ###
    valid_img_gen = train_img_data_gen.flow_from_directory(os.path.join(training_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs, 
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED,
                                                       subset='validation')  
    
    valid_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(training_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         shuffle=True,
                                                         color_mode='rgba',
                                                         interpolation='bilinear',
                                                         seed=SEED,
                                                         subset='validation')
    
    #print (len(valid_img_gen))

                                         
    train_gen = zip(train_img_gen, train_mask_gen)
    valid_gen = zip(valid_img_gen, valid_mask_gen)

    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 4]))

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 4]))

    
    def prepare_target(x_, y_):
        y_ = tf.cast(y_, tf.int32)
        return x_, y_

    train_dataset = train_dataset.map(prepare_target)
    train_dataset = train_dataset.repeat()

    valid_dataset = valid_dataset.map(prepare_target)
    valid_dataset = valid_dataset.repeat()
    
    
    """
    iterator = iter(train_dataset)
    
    for _ in range(5):
        images, masks = next(iterator)    
        for i in range(batch_size):
            image = images[i]   # First element
            image = image * 255  # denormalize
            image = tf.dtypes.cast(image, tf.int32) #Why this??

            mask = masks[i]   # First element
            mask = mask * 255  # denormalize

            image = image.numpy()
            mask = mask.numpy()
            util.show_imgs((image,
                                mask[:,:,0], mask[:,:,1],
                                mask[:,:,2], mask[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))
            
        
    
    iterator = iter(valid_dataset)
  
    for _ in range(5):
        images, masks = next(iterator)    
        for i in range(batch_size):
            image = images[i]   # First element
            image = image * 255  # denormalize
            image = tf.dtypes.cast(image, tf.int32) #Why this??

            mask = masks[i]   # First element
            mask = mask * 255  # denormalize

            image = image.numpy()
            mask = mask.numpy()
            util.show_imgs((image,
                                mask[:,:,0], mask[:,:,1],
                                mask[:,:,2], mask[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))

    """
    
    
    return train_dataset,valid_dataset



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









def generate_masks (img_h, img_w, batch_size, preprocess_type='None'):
    training_dir = 'Severstal_Dataset'

    tr = pd.read_csv(training_dir + '/train.csv')
    
    df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
    print(len(df_train))

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

        mask = mask*255
            
        return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


    
    def keras_generator(imgshape=(256,1600)):
        mask_dataset = {}

        for i in range(len(df_train)):      
            filename = df_train['ImageId_ClassId'].iloc[i].split('_')[0]
            empty_mask = np.zeros(shape=(256,1600,4),dtype=np.uint8)
            mask_dataset[filename] = empty_mask

        """
        print (len(mask_dataset))
        for i in range(len(df_train)):        
            filename = df_train['ImageId_ClassId'].iloc[i].split('_')[0]
            classID = df_train['ImageId_ClassId'].iloc[i].split('_')[1]        

            mask = rle2mask(df_train['EncodedPixels'].iloc[i], imgshape)

            img = mask_dataset[filename]
            img[:,:,(int(classID)-1)] = mask
            mask_dataset[filename] = img

        
        for i in range(len(df_train)):       
            filename = df_train['ImageId_ClassId'].iloc[i].split('_')[0] 
            classID = df_train['ImageId_ClassId'].iloc[i].split('_')[1]
            print (filename + ' ' + classID)
            name = os.path.splitext(filename)[0]

            cv2.imwrite("Severstal_Dataset\\train_images\\masks\\" + name + ".png", mask_dataset[filename])
        """
        
        for i in range(30):    
            filename = df_train['ImageId_ClassId'].iloc[i].split('_')[0] 
            classID = df_train['ImageId_ClassId'].iloc[i].split('_')[1]
            name = os.path.splitext(filename)[0]
            
            img = cv2.imread( training_dir + '\\train_images\\imgs\\' + filename )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            mask = cv2.imread( training_dir + '\\train_images\\masks\\' + name + ".png", -1)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

            util.show_imgs((img,
                            mask[:,:,0], mask[:,:,1],
                            mask[:,:,2], mask[:,:,3]),
                           (filename,'1','2','3','4'),('','','','',''))

    keras_generator() 
    
    return None



from os import listdir
from os.path import isfile, join
def keep_train_imgs ():
    mask_path = 'Severstal_Dataset\\train_images\\masks\\imgs'
    mask_files = [f for f in listdir(mask_path) if isfile(join(mask_path, f))]

    img_path = 'Severstal_Dataset\\train_images\\images_all\\imgs'
    dest_path = 'Severstal_Dataset\\train_images\\images\\imgs'
    for i in range(len(mask_files)):
        name = os.path.splitext(mask_files[i])[0]
        img = cv2.imread(img_path + '\\' + name + '.jpg')
        
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        #plt.imshow(img)
        #plt.show()

        cv2.imwrite(dest_path + '\\' + name + '.jpg', img)