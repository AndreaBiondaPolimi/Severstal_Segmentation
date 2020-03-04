import keras
import keras.losses
from keras.models import save_model, load_model
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import keras.backend as K
import tensorflow as tf
import math
import sys
import struct
import cv2 
#from sklearn import svm
from keras.models import Sequential, Model
import matplotlib as mpl



def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

def sum_of_square (y_true,y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))

def mse_loss_topK (y_true,y_pred):
    val = K.square(y_true - y_pred)
    val = K.flatten(val)
    val_topk, ind_topk = tf.nn.top_k(val, k=13107) # 5% of (512x512) = 13107 (hardcoded for a while)
    return K.mean(val_topk)   




def load_train_dataset():
    path_train_orig = "ML\\train\\orig"
    path_train_bin = "ML\\train\\bin"
    path_train_view = "ML\\train\\view"

    #Load from file train images
    train_orig = loadImages(path_train_orig,100)
    train_bin = loadImages(path_train_bin,100)  
    train_view = loadImages(path_train_view,100)  

    #Apply image reshaping
    train_orig = np.reshape(train_orig, (len(train_orig),512,512,1))
    train_bin = np.reshape(train_bin, (len(train_bin),512,512,1))
    train_view = np.reshape(train_view, (len(train_view),512,512,1))
    
    return train_orig, train_bin, train_view

def load_test_dataset ():
    path_test_orig = "ML\\test\\orig"
    path_test_bin = "ML\\test\\bin"
    path_test_view = "ML\\test\\view"
    
    #Load from file test images
    test_orig = loadImages(path_test_orig,100)
    test_bin = loadImages(path_test_bin,100)  
    test_view = loadImages(path_test_view,100)  

    #Apply image reshaping
    test_orig = np.reshape(test_orig, (len(test_orig),512,512,1))
    test_bin = np.reshape(test_bin, (len(test_bin),512,512,1))
    test_view = np.reshape(test_view, (len(test_view),512,512,1))
    
    return test_orig, test_bin, test_view



###     LOAD/SAVE FILES     ###
def loadRIDImages(path):
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint16)
        img = np.reshape(data, (512, 512))
        return img

def loadImages (path, maxNumber):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = sorted(files)   
    array = []
    for f in files:
        #img_arr = plt.imread(path + "\\" + f) #Read Tiff images
        img_arr = loadRIDImages(path + "\\" + f) #Read RID images
        array.append(img_arr)
    imgs = np.array(array[0:maxNumber])  # transformed to a numpy array

    for i in range(len(imgs)):
        imgs[i] = cut_image(imgs[i], [20, 20, 472, 472])

    
    """
    for i in range(len(imgs)):
        imgs[i] = cut_image(imgs[i], [20, 20, 472, 472])
        plt.imshow(imgs[i])
        plt.show()
    """
    
    return imgs

def cut_image (img, ROI):
    np_mask =  np.zeros((512,512),dtype=int)
    np_mask[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])] = img[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]

    return np_mask

def saveImgs(autoencoder,scrap_imgs,dirName):
    path = "AutoencoderModels\\" + dirName
    os.mkdir(path)
    
    for i in range(0,8):
        temp1 = autoencoder.predict(np.reshape(scrap_imgs[i],(1,512,512,1)))
        temp1 =  np.reshape(temp1,(512,512,1)) - scrap_imgs[i] 
        
        plt.imshow(np.reshape(temp1,(512,512)))
        plt.savefig(path + '\\Img' + str(i))




###     IMAGE PREPARATION     ###
def normaliazation (img,nbit,type):
    depth = (math.pow(2, nbit)-1)
    if type == 0:
        img = img / depth
    else:
        img = (img - (depth/2) ) / (depth/2)
    return img

def denormaliazation (img,nbit,type):
    depth = (math.pow(2, nbit)-1)
    if type == 0:
        img = img * depth
    else:
        img = (img * (depth/2)) + (depth/2)
    return img



def data_augmentation (x_train):
    x_copy = x_train
    for i in range(1,2):
        x_train = np.vstack((x_train,x_copy))
    return x_train



def align(good_images,scrap_images,norm_type):
    ref = good_images[0]
    for i in range (0,len(good_images)):
        good_images[i] = imageAlignement(good_images[i],ref,i,norm_type,'GoodAlign')
    for i in range (0,len(scrap_images)):
        scrap_images[i] = imageAlignement(scrap_images[i],ref,i,norm_type,'ScrapAlign')
    return good_images,scrap_images


def imageAlignement(img,ref,i,norm_type,path):
    img_gray = np.reshape((img/256).astype('uint8'),(512,512))
    template = np.reshape((ref/256).astype('uint8'),(512,512))
    template1 = template[32:450,32:450] #Hardcoded beacuse works

    result = cv2.matchTemplate(img_gray,template1, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h,w = template1.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    padding_w = int((512-w)/2)
    padding_h = int((512-h)/2)
    ret = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    ret = cv2.copyMakeBorder(ret,padding_h,padding_h,padding_w,padding_w,cv2.BORDER_CONSTANT)

    save = denormaliazation(ret,16,norm_type)
    save = save.astype(np.uint16)

    cv2.imwrite("D:\\usw-andreab\\Desktop\\TensorProjects\\AnomalyDetectAutoencoder\\AnomalyDetectAutoencoder\\ML\\" + path + "\\good" + str(i) + ".tif",save)
    return ret

def crop (image,crop_dim):
    tot_dim = len(image[0])
    n_parts = int((tot_dim/crop_dim) * (tot_dim/crop_dim)) 
    samples = np.zeros((n_parts,crop_dim,crop_dim,1))

    h_x = 0
    part = 0
    while h_x < tot_dim:
        h_y=0
        while h_y < tot_dim:
            crop_img = image[h_x:h_x+crop_dim, h_y:h_y+crop_dim]
            samples[part] = crop_img
            h_y = h_y + crop_dim
            part=part+1
        h_x = h_x + crop_dim
    
    return samples


def mean_refactor(images):
    ref = np.mean(images[0])
    for i in range (len(images)):
        images[i] = images[i] + (ref-np.mean(images[i]))
    return images

def define_ROI (img):
    ROI = cv2.selectROI(img, False)
    """
    img = img[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
    cv2.imshow('',img)
    cv2.waitKey()
    """
    return ROI




###     MODEL TESTING     ###
def load_target_image (idx):
    path = "ML\\ScrapBin"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    array = []
    for f in files:
        #img_arr = plt.imread(path + "\\" + f) #Read Tiff images
        img_arr = loadRIDImages(path + "\\" + f) #Read RID images
        array.append(img_arr)
    imgs = np.array(array[0:10])

    return imgs[idx]


def show_test_results (out_img, target_img, original_img):
    precision, recall = get_statistic (out_img,target_img)

    msg = "precision: " + str(precision) + " recall:" + str(recall)

    show_imgs ((original_img,out_img,target_img), ("original", "output","target"), (msg, "", ""))


def prepare_diff_img (diff_img, shape, ROI):
    diff_img = ((diff_img - diff_img.min()) / (diff_img.max() - diff_img.min())).astype(np.float) # normalization
    diff_img = np.reshape(diff_img,(shape,shape))[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
    diff_img = diff_img / diff_img.max()

    return diff_img


def show_imgs (imgs, titles, messages):
    n_img = len(imgs)
    _ , axarr = plt.subplots(n_img)
    
    for i in range (n_img):
        if i==0:
            axarr[i].imshow(np.reshape(imgs[i],(256,1600,3)))
        else:
            axarr[i].imshow(np.reshape(imgs[i],(256,1600)))
        axarr[i].set_title(titles[i])
        axarr[i].text(0, 200, messages[i])
        axarr[i].axis('off')

    fig = plt.gcf()
    fig.set_size_inches(14, 8, forward=True)
    plt.tight_layout()

    plt.show()


def treshold_img (diff_img, tresh_val):
    diff_img[diff_img < tresh_val] = 0.
    diff_img[diff_img >= tresh_val] = 1.
    return diff_img


def get_statistic (out_img, target_img):
    out_img[out_img>=0.2] = 1
    out_img[out_img<0.2] =  0

    mult_img = np.multiply(out_img, target_img)
    diff_img = out_img - target_img
    diff_img_fp = diff_img*(diff_img>0)             
    diff_img_fn = diff_img*(diff_img<0)          

    true_pos = np.count_nonzero(mult_img)
    false_pos = np.count_nonzero(diff_img_fp)
    false_neg = np.count_nonzero(diff_img_fn)

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0

    return round(precision,4), round(recall,4)