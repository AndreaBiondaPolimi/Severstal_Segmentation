import DataLoader as dl
import Segmentator as seg
import sys
import tensorflow as tf
import DataGenerator as dg


def start ():

    img_h=128
    img_w=800

    #preprocess_type='resnet34'; batch_size=16
    preprocess_type='efficientnetb4'; batch_size=4

    #train, valid = dl.load_dataset(img_h, img_w, batch_size, preprocess_type)
    
    train, valid = dg.load_dataset(preprocess_type, batch_size)
    
    
    
    model = seg.get_segmentation_model(input_size=(img_h,img_w,3),
                                       preprocess_type=preprocess_type,
                                       pretrained_weights=None)

    seg.train(model, train, valid, 50)

    #dl.test_model(model, True, img_w, img_h, preprocess_type)

start()