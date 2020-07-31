import Segmentator as seg
import sys
import tensorflow as tf
import DataGenerator as dg
import Classificator as cla

#preprocess_type='resnet34'
#preprocess_type='resnet50'
preprocess_type='efficientnetb3'

#activation = 'sigmoid'
activation = 'softmax'

use_defective_only = True
#use_defective_only = False

def segment ():
    train, valid = dg.load_dataset_segmentation(preprocess_type, use_defective_only, activation)
    
    model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                pretrained_weights=None, activation=activation)

    seg.train(model, train, valid, 100)


def classify ():
    train, valid = dg.load_dataset_classification(preprocess_type)

    model = cla.get_classification_model(preprocess_type=preprocess_type,
                pretrained_weights='check_acc23_0.91_resnet50_lr0.00012.h5')

    cla.train(model, train, valid, 50)


def test ():
    cla_model = cla.get_classification_model(preprocess_type='resnet34',
                        pretrained_weights='check_acc_0.55_resnet34.h5')

    seg_model = seg.get_segmentation_model(preprocess_type='efficientnetb3',
                        pretrained_weights='check_val_dice28_0.59_3x10-6.h5', 
                        activation=activation)

    dg.test_model(seg_model, cla_model, seg_preprocess_type='efficientnetb3', cls_preprocess_type='resnet50',
    activation=activation)
    
    #dg.fulltest_classification_model (cla_model, cls_preprocess_type='resnet50')

def hyperparameters_Search ():
    _, valid = dg.load_dataset_segmentation(preprocess_type, use_defective_only, activation)

    seg_model = seg.get_segmentation_model(preprocess_type=preprocess_type,
                        pretrained_weights='check_val_dice28_0.59_3x10-6.h5', 
                        activation=activation)

    seg.search_segmentation_treshold(valid, seg_model, activation, preprocess_type)

#segment()
#classify()
#test()
hyperparameters_Search()