import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

#Show image
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
    for _ in range (5): #Try n times to get the random crop before rx/lx choice
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