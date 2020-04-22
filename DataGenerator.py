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

class DataGenerator(keras.utils.Sequence):
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
        y = np.empty((self.batch_size,self.img_h,self.img_w,4),dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(self.data_path + f).resize((self.img_w,self.img_h)) 
            for j in range(4):
                y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]], self.img_h, self.img_w)
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
            y[i] = apply_affine_transform(y[i], theta=theta, tx=tx, ty=ty, zx=z, zy=z,
                                         fill_mode=fill_mode)

            X[i] = datagen.apply_transform(x=X[i], 
                            transform_parameters={'brightness':br, 'flip_horizontal':f_h, 'flip_vertical':f_v})

            y[i] = datagen.apply_transform(x=y[i], 
                            transform_parameters={'flip_horizontal':f_h, 'flip_vertical':f_v})
        
        
        return X, y



# https://www.kaggle.com/titericz/building-and-visualizing-masks
def rle2maskResize(rle, img_h, img_w):
    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return np.zeros((img_h,img_w) ,dtype=np.uint8)
    
    height= img_h
    width = img_w
    mask= np.zeros( width*height ,dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1

    #ret = mask.reshape( (height,width), order='F' )[::2,::2]
    ret = mask.reshape( (height,width), order='F' )
    
    return ret

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


def restructure_data_frame(path):
    train = pd.read_csv(path)

    # RESTRUCTURE TRAIN DATAFRAME
    train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')

    train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
    train2['e1'] = train['EncodedPixels'][::4].values
    train2['e2'] = train['EncodedPixels'][1::4].values
    train2['e3'] = train['EncodedPixels'][2::4].values
    train2['e4'] = train['EncodedPixels'][3::4].values
    train2.reset_index(inplace=True,drop=True)
    train2.fillna('',inplace=True)
    train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values
    train2.head()

    return train2



import segmentation_models as sm
def load_dataset (img_h, img_w, preprocess_type, batch_size=16):
    train2 = restructure_data_frame('Severstal_Dataset\\train.csv')
    idx = int(0.8*len(train2)); print()
    preprocess = sm.get_preprocessing(preprocess_type)

    train_batches =  DataGenerator(train2.iloc[:idx], img_h=img_h, img_w = img_w, shuffle=True, preprocess=preprocess, batch_size=batch_size,
                                    width_shift_range=20, height_shift_range=20, zoom_range=0.4, 
                                    flip_h=True, flip_v=True, brightness=0.4)

    valid_batches = DataGenerator(train2.iloc[idx:],img_h=img_h, img_w = img_w, preprocess=preprocess, batch_size=batch_size)
    
    """
    iterator = iter(train_batches)
    for _ in range(20):
        images, masks = next(iterator)
        masks = masks * 255

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]
            print (np.min(image))
            print (np.min(mask))
            print (np.max(image))
            print (np.max(mask))

            util.show_imgs((image,
                            mask[:,:,0], mask[:,:,1],
                            mask[:,:,2], mask[:,:,3]),
                            ('orig','1','2','3','4'),('','','','',''))
    """

    return train_batches, valid_batches 


from tqdm import tqdm
def test_model(model, img_h, img_w, preprocess_type):
    train2 = restructure_data_frame('Severstal_Dataset\\test.csv')
    preprocess = sm.get_preprocessing(preprocess_type)
    
    test_batches = DataGenerator(train2, img_h=img_h, img_w=img_w, preprocess=preprocess, batch_size=1, subset='test')

    n_samples = 500
    dice_res = 0
    iterator = iter(test_batches)
    for _ in tqdm(range(n_samples)):
        images, masks = next(iterator)

        for i in range(len(images)):
            image = images[i].astype(np.int16)
            mask = masks[i]

            #mask = mask * 255
            #util.show_imgs((image,
                            #mask[:,:,0], mask[:,:,1],
                            #mask[:,:,2], mask[:,:,3]),
                            #('orig','1','2','3','4'),('','','','',''))


            res = model.predict(np.reshape(image,(1,img_h,img_w,3)))
            res = np.reshape(res,(img_h,img_w,4))
            res[np.where(res < 0.5)] = 0
            res[np.where(res >= 0.5)] = 1

            dice_res += dice_coef(mask.astype(np.uint8), res.astype(np.uint8))

            #res = res * 255
            #util.show_imgs((image,
                            #res[:,:,0], res[:,:,1],
                            #res[:,:,2], res[:,:,3]),
                            #('orig','1','2','3','4'),('','','','',''))

    print (dice_res/n_samples)


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