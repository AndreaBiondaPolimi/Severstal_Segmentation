import Segmentator as seg
import sys
import tensorflow as tf
import DataGenerator as dg
import Classificator as cla

def segment ():
    preprocess_type='resnet34'
    #preprocess_type='efficientnetb4'; batch_size=1

    train, valid = dg.load_dataset_segmentation(preprocess_type)
    
    
    model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_val_dice37.h5')

    #seg.train(model, train, valid, 100)

    dg.test_model(model, preprocess_type)




def classify ():
    preprocess_type='resnet34'

    train, valid = dg.load_dataset_classification(preprocess_type)

    model = cla.get_classification_model(preprocess_type=preprocess_type,
                                       pretrained_weights=None)

    cla.train(model, train, valid, 50)

    #dg.test_model(model, img_h, img_w, preprocess_type)


#segment()
classify()