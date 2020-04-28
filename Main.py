import Segmentator as seg
import sys
import tensorflow as tf
import DataGenerator as dg
import Classificator as cla

img_h=128
img_w=800

def segment ():
    preprocess_type='resnet34'; batch_size=4
    #preprocess_type='efficientnetb4'; batch_size=1

    train, valid = dg.load_dataset_segmentation(img_h, img_w, preprocess_type, batch_size)
    
    
    model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                                       pretrained_weights=None)

    seg.train(model, train, valid, 50)

    #dg.test_model(model, img_h, img_w, preprocess_type)




def classify ():
    preprocess_type='resnet34'; batch_size=4

    train, valid = dg.load_dataset_classification(img_h, img_w, preprocess_type, batch_size)

    model = cla.get_classification_model(input_size=(img_h,img_w,3),
                                       preprocess_type=preprocess_type,
                                       pretrained_weights=None)

    seg.train(model, train, valid, 50)

    #dg.test_model(model, img_h, img_w, preprocess_type)


segment()
#classify()