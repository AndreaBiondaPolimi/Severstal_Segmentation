import DataLoader as dl
import Segmentator as seg
import sys
import tensorflow as tf



def start ():

    img_h=256
    img_w=1600
    batch_size=1

    #dl.generate_masks(img_h, img_w, batch_size)
    #dl.keep_train_imgs()

    train, valid = dl.load_dataset(img_h, img_w, batch_size)
    
    model = seg.get_segmentation_model()

    seg.train(model, train, valid, 50)

    #dl.test_model(model, True, img_h, img_w)

start()