import Segmentator as seg
import sys
import tensorflow as tf
import DataGenerator as dg
import Classificator as cla

#preprocess_type='resnet34'
preprocess_type='efficientnetb3'

def segment ():
    train, valid = dg.load_dataset_segmentation(preprocess_type)
    
    model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_val_dice06_0.6.h5')

    seg.train(model, train, valid, 100)


def classify ():
    train, valid = dg.load_dataset_classification(preprocess_type)

    model = cla.get_classification_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_acc32.h5')

    cla.train(model, train, valid, 50)


def test ():
    cla_model = cla.get_classification_model(preprocess_type='resnet34',
                                       pretrained_weights='check_acc32.h5')

    seg_model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                                       pretrained_weights='check_val_dice06_0.6.h5')

    dg.test_model(seg_model, cla_model, seg_preprocess_type=preprocess_type, cls_preprocess_type='resnet34')

#segment()
#classify()
test()