import json
from data_utils.data_loader import image_segmentation_generator, verify_segmentation_dataset
import glob
import six
from os import path
import cv2
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from models.fcn import fcn_32
from train import train, find_latest_checkpoint



def predict():
    out = model.predict_segmentation(
        inp="drive/My Drive/tmp/voc_jpeg/7indexmap.jpg",
        out_fname="/tmp/out.png"
    )
    plt.imshow(out),plt.show()

    num_classes = num_cls
    plt.imshow(out, vmin=0, vmax=num_classes-1, cmap='jet')
    plt.show()


def load_weight():
    num_cls = 51
    input_height = 416
    input_width = 608

    # load model
    model = fcn_32(n_classes = num_cls, input_height=input_height, input_width=input_width)
    model.summary()
    model.load_weights('/home/ubuntu/coco/tmp/22.h5')


    out = model.predict_segmentation(
        inp="/home/ubuntu/coco/dataset1/images_prepped_test/0016E5_07959.png",
        out_fname="/tmp/out.png"
    )
    print(model.evaluate_segmentation(inp_images_dir="/home/ubuntu/coco/dataset1/images_prepped_test/"  , annotations_dir="/home/ubuntu/coco/dataset1/annotations_prepped_test/" ) )

    import matplotlib.pyplot as plt
    plt.imshow(out)
    
if __name__ == '__main__':
    num_cls = 51
    input_height = 416
    input_width = 608

    # load model
    model = fcn_32(n_classes = num_cls, input_height=input_height, input_width=input_width)
    model.summary()

    # train
    directory = 'tmp'
    os.makedirs(directory, exist_ok=True)

    model.train(
        train_images =  "dataset/images_train",
        train_annotations = "dataset/annotations_train",
        verify_dataset=True,
        validate=True,
        val_images="dataset/images_test",
        val_annotations="dataset/annotations_test",
        val_batch_size=10,
        val_steps_per_epoch=100,
        checkpoints_path = directory, epochs=30
    )




