import Segmentator as seg
import sys
import tensorflow as tf
import DataGenerator as dg
import Classificator as cla
import argparse
from utils import Settings

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action="store", help='Action to perform: train or test', dest="action", type=check_action, required=True)
    parser.add_argument('-t', action="store", help='Target of the action: segmentation, classification or pipeline', dest="target", type=check_target, required=True)
    parser.add_argument('-s', action="store", help='Settings file name, by default is Settings.ini', dest="settings", default='Settings.ini')
    args = parser.parse_args()

    print (args.action)
    print (args.target)
    print (args.settings)

    config = Settings(args.settings, args.action, args.target)

    if (args.action == 'train'):
        if (args.target == 'segmentation'):
            segment(config.segmentation_model, config.segmentation_activation, config.loss,
                        config.epochs, config.use_balanced_batch, config.shape, config.segmentation_weights)
        elif (args.target == 'classification'):
            classify(config.classification_model, config.epochs, config.classification_weights)
        else:
            raise "Target type error"

    else:
        if (args.target == 'segmentation'):
            test_pipeline (config.classification_model, config.classification_weights, config.segmentation_model, config.segmentation_weights,
                                config.segmentation_activation, config.verbose, 0.0)
        elif (args.target == 'pipeline'):
            test_pipeline (config.classification_model, config.classification_weights, config.segmentation_model, config.segmentation_weights,
                                config.segmentation_activation, config.verbose, 0.5)
        else:
            test_classification (config.classification_model, config.classification_weights)


def check_action(value):
    if value != "train" and value != "test":
        raise argparse.ArgumentTypeError("Invalid action argument")
    return value

def check_target(value):
    if value != "segmentation" and value != "classification" and value != "pipeline":
        raise argparse.ArgumentTypeError("Invalid target argument")
    return value


def segment (preprocess_type, activation, loss, epochs, use_balanced_batch, shape, pretrained_weights):
    train, valid = dg.load_dataset_segmentation(preprocess_type, activation, use_balanced_batch, shape)
    model = seg.get_segmentation_model(preprocess_type=preprocess_type, pretrained_weights=pretrained_weights, activation=activation)
    seg.train(model, train, valid, epochs)

def classify (preprocess_type, epochs, pretrained_weights):
    train, valid = dg.load_dataset_classification(preprocess_type)
    model = cla.get_classification_model(preprocess_type=preprocess_type, pretrained_weights=pretrained_weights)
    cla.train(model, train, valid, epochs)


def test_pipeline (class_preprocess_type, class_weights, seg_preprocess_type, seg_weights, seg_activation, verbose, cls_treshold):
    cla_model = cla.get_classification_model(preprocess_type=class_preprocess_type, pretrained_weights=class_weights)
    seg_model = seg.get_segmentation_model(preprocess_type=seg_preprocess_type, pretrained_weights=seg_weights, activation=seg_activation)
    dg.test_pipeline(seg_model, cla_model, seg_preprocess_type=seg_preprocess_type, cls_preprocess_type=class_preprocess_type, 
                        cls_treshold = cls_treshold, activation=seg_activation, verbose=verbose)

def test_classification (class_preprocess_type, class_weights):
    cla_model = cla.get_classification_model(preprocess_type=class_preprocess_type, pretrained_weights=class_weights)
    dg.test_classification (cla_model, class_preprocess_type)


if __name__ == "__main__":
    main()