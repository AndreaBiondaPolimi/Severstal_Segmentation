import DataLoader as dl
import Segmentator as seg
import sys
import tensorflow as tf



def start ():

    img_h=512
    img_w=512
    batch_size=2


    #train = dl.load_dataset(img_h, img_w, batch_size)
    train = dl.load_dataset_v2(img_h, img_w, batch_size)
    
    #model = seg.get_segmentation_model(model_type='class')

    #seg.train(model, train, 50)

    #dl.test_model(model, True, img_h, img_w)

start()