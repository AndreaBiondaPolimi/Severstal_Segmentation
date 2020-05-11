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
                                       pretrained_weights=None)

    seg.train(model, train, valid, 100)


def classify ():
    preprocess_type='resnet34'

    train, valid = dg.load_dataset_classification(preprocess_type)

    model = cla.get_classification_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_acc32.h5')

    cla.train(model, train, valid, 50)


def test ():
    preprocess_type='resnet34'

    cla_model = cla.get_classification_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_acc32.h5')

    seg_model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_val_dice60_0.5023.h5')

    dg.test_model(seg_model, cla_model, preprocess_type)

segment()
#classify()
#test()