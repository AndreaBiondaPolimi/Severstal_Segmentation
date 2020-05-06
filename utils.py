import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

