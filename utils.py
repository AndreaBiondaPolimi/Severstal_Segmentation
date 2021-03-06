import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

#Show images
def show_imgs (imgs, titles, messages):
    n_img = len(imgs)
    _ , axarr = plt.subplots(n_img)
    
    for i in range (n_img):
        if i==0:
            axarr[i].imshow(imgs[i])
        else:
            axarr[i].imshow(imgs[i])
        axarr[i].set_title(titles[i])
        axarr[i].text(0, 200, messages[i])
        axarr[i].axis('off')

    fig = plt.gcf()
    fig.set_size_inches(14, 8, forward=True)
    plt.tight_layout()

    plt.show()

def show_img_over_def (img, mask):
    plt.figure(figsize=(14,2)) #20,18
    extra = '  has defect'
    for j in range(4):
        msk = mask[:,:,j]
        msk = mask2pad(msk,pad=3)
        msk = mask2contour(msk,width=4)
        if np.sum(msk)!=0: extra += ' '+str(j+1)
        if j==0: # yellow
            img[msk==1,0] = 235 
            img[msk==1,1] = 235
        elif j==1: img[msk==1,1] = 210 # green
        elif j==2: img[msk==1,2] = 255 # blue
        elif j==3: # magenta
            img[msk==1,0] = 255
            img[msk==1,2] = 255

    yellow_patch = mpatches.Patch(color='yellow', label='Defect 1')
    green_patch = mpatches.Patch(color='green', label='Defect 2')
    blue_patch = mpatches.Patch(color='blue', label='Defect 3')
    red_patch = mpatches.Patch(color='magenta', label='Defect 4')
    plt.legend(handles=[yellow_patch, green_patch, blue_patch, red_patch])

    plt.axis('off') 
    plt.imshow(img)
    
    plt.show()

def show_img_and_def (images, titles):
    _ , axarr = plt.subplots(len(images))

    for i in range (len(images)):
        img = images[i]
        title = titles[i]
        if i == 0:
            axarr[i].imshow(img)
        else:
            mask = mask2view(img)
            axarr[i].imshow(mask)
        
        axarr[i].set_title(title)
        

    fig = plt.gcf()
    fig.set_size_inches(14, 8, forward=True)
    yellow_patch = mpatches.Patch(color='yellow', label='Defect 1')
    green_patch = mpatches.Patch(color='green', label='Defect 2')
    blue_patch = mpatches.Patch(color='blue', label='Defect 3')
    red_patch = mpatches.Patch(color='red', label='Defect 4')
    plt.legend(handles=[yellow_patch, green_patch, blue_patch, red_patch])
    plt.tight_layout()

    plt.show()


#Dataframe restructure
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


def get_random_split (train2, p = 0.8):
    valid_idxs = np.random.choice( a=[False, True], size=(len(train2)), p=[p, 1-p])
    train_idxs = np.logical_not(valid_idxs)
    
    return train_idxs, valid_idxs


def get_defective_data_frame (train2):
    return train2.loc[train2['count'] > 0]



#Mask creation
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

def mask2Background (mask):
    background = np.zeros((256,1600),dtype=np.int8)
    for j in range(4):
        background = np.logical_or(background, mask[:,:,j])
    background = np.logical_not(background)

    return background

#Mask visualization
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

def mask2view(mask):
    view = np.zeros((mask.shape[0],mask.shape[1],3), dtype=np.uint8)
    for j in range(4):
        layer = mask[:,:,j]
        if (j==0):  #yellow
            view[:,:,0] = view[:,:,0] + (layer.astype(np.uint8))*255
            view[:,:,1] = view[:,:,1] + (layer.astype(np.uint8))*255
            view[:,:,2] = view[:,:,2] + (layer.astype(np.uint8))*0
        
        if (j==1):  #yellow
            view[:,:,0] = view[:,:,0] + (layer.astype(np.uint8))*0
            view[:,:,1] = view[:,:,1] + (layer.astype(np.uint8))*255
            view[:,:,2] = view[:,:,2] + (layer.astype(np.uint8))*0
        
        if (j==2):  #yellow
            view[:,:,0] = view[:,:,0] + (layer.astype(np.uint8))*0
            view[:,:,1] = view[:,:,1] + (layer.astype(np.uint8))*0
            view[:,:,2] = view[:,:,2] + (layer.astype(np.uint8))*255

        if (j==3):  #yellow
            view[:,:,0] = view[:,:,0] + (layer.astype(np.uint8))*255
            view[:,:,1] = view[:,:,1] + (layer.astype(np.uint8))*0
            view[:,:,2] = view[:,:,2] + (layer.astype(np.uint8))*0

    return view


#Data augmentation with albumentations
def get_augmentation(augm_param):
    affine_augmentation = Compose([
        Flip(p=augm_param['flip_prob']),
        ShiftScaleRotate(shift_limit=augm_param['shift_limit'], rotate_limit=augm_param['rotate_limit'], p=augm_param['shift_rot_prob']),
    ], p=1)

    color_augmentation = Compose([
        OneOf([
            RandomContrast(augm_param['contrast_limit']),
            RandomBrightness(augm_param['brightness_limit']),
        ], p=augm_param['contr_bright_prob']),
    ], p=1)

    return affine_augmentation, color_augmentation

def augment(aug, image, mask=None):
    if (mask is None):
        data = {"image": image}
        result = aug(**data)
        return result['image']
    
    data = {"image": image, "mask": mask}
    result = aug(**data)
    return result['image'], result['mask']


#Random crop generatiom
def random_crop(img, mask, random_crop_indexes):
    (dx, dy), (x,y) = random_crop_indexes

    img_cropped = img[y:(y+dy), x:(x+dx), :]
    if (mask is not None):
        mask_cropped = mask[y:(y+dy), x:(x+dx)]
    else:
        mask_cropped = None
    return (img_cropped, mask_cropped)

def get_random_crop_indexes(original_image_size, random_crop_size, img, mask):
    height, width = original_image_size
    dy, dx = random_crop_size

    #Try to get the random crop that contains some of the defect, if present
    if (mask is not None) and (np.count_nonzero(mask) > 0):
        for _ in range (30): #Try n times to get the random crop before give up
            x = np.random.randint(0, width - dx + 1)
            y = np.random.randint(0, height - dy + 1)
            if (not is_total_black(mask, x, y, dx, dy, 0, 40)):
                return ((dx, dy), (x,y))
        

    #Try to get the random crop that does not show full black image, if defect not present
    for _ in range (20): #Try n times to get the random crop before rx/lx choice
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        if (not is_total_black(img, x, y, dx, dy)):
            return ((dx, dy), (x,y))

    #usually steel is on the right or on the left when background is present
    #Try with left crop
    x = 0
    y = np.random.randint(0, height - dy + 1)
    if (not is_total_black(img, x, y, dx, dy)):
        return ((dx, dy), (x,y))

    #Try with right crop
    x = width - dx - 1
    y = np.random.randint(0, height - dy + 1)

    return ((dx, dy), (x,y))
   
def is_total_black(img, x, y, dx, dy, treshold=30, quantity=0):
    cropped_img = img[y:(y+dy), x:(x+dx), :].copy()

    cropped_img[cropped_img < treshold] = 0
    if (np.count_nonzero(cropped_img) > quantity):
        return False
    return True




import configparser
class Settings:
    def __init__(self, path_ini, action, target):
        self.config = configparser.ConfigParser()
        self.config.read(path_ini)

        if (target == 'segmentation'):
            self.segmentation_model = self.config['SEGMENTATION_TRAINING']['model']
            self.segmentation_activation = self.config['SEGMENTATION_TRAINING']['activation']
            self.use_balanced_batch = bool(self.config.getboolean('SEGMENTATION_TRAINING','use_balanced_batch'))
            self.loss = self.config['SEGMENTATION_TRAINING']['loss']
            self.epochs = int(self.config['SEGMENTATION_TRAINING']['epochs'])
            self.shape = list(eval(self.config['SEGMENTATION_TRAINING']['shape'])) 
            self.segmentation_weights = self.config['SEGMENTATION_TRAINING']['pretrained_weights']
            if (self.segmentation_weights == 'None'):
                self.segmentation_weights = None
            
        elif (target == 'classification'):
            self.classification_model = self.config['CLASSIFICATION_TRAINING']['model']
            self.epochs = int(self.config['CLASSIFICATION_TRAINING']['epochs'])
            self.classification_weights = self.config['CLASSIFICATION_TRAINING']['pretrained_weights']
            if (self.classification_weights == 'None'):
                self.classification_weights = None


        if (action == 'test'):
            self.segmentation_model = self.config['TEST']['segmentation_model']
            self.classification_model = self.config['TEST']['classification_model']
            self.segmentation_weights = self.config['TEST']['segmentation_weights']
            self.classification_weights = self.config['TEST']['classification_weights']
            self.segmentation_activation = self.config['TEST']['segmentation_activation']
            self.verbose = bool(self.config.getboolean('TEST','verbose'))