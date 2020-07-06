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
from keras.models import *
from keras.layers import *
from models.model_utils import get_segmentation_model
from models.config import IMAGE_ORDERING


def create_resized_img(img, H=608, W=416):
      resize_img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
      #plt.imshow(resize_img),plt.show()
      h, w, _ = np.shape(resize_img)
      base_indexmap = np.zeros([h, w, 3])
      return resize_img, base_indexmap



def create_finetune_model(model, n_classes = 10):
  inputs_ = model.inputs
  dense = model.get_layer(index=-5).output
  o = (Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal',
                data_format=IMAGE_ORDERING))(dense)
  o = Conv2DTranspose(n_classes, kernel_size=(64, 64),  strides=(32, 32), use_bias=False,  data_format=IMAGE_ORDERING)(o)
  fine_model = get_segmentation_model(inputs_, o)
  fine_model.summary()
  return fine_model

def load_pretrain():
    num_cls = 51
    input_height = 416
    input_width = 608

    # load model
    model = fcn_32(n_classes = num_cls, input_height=input_height, input_width=input_width)
    model.load_weights('/content/drive/My Drive/fcn32_weight.h5')
    return model

def finetune_train():
    model = load_pretrain()
    finetune_model = create_finetune_model(model, n_classes = 11)
    # train
    directory = 'tmp'
    os.makedirs(directory, exist_ok=True)

    finetune_model.train(
            train_images = "/content/drive/My Drive/resized_train",
            train_annotations = "/content/drive/My Drive/indexmap",
            checkpoints_path = directory, epochs=30
        )

    path = '/content/drive/My Drive/seg_train_images/train_0355.jpg'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, _ = create_resized_img(img)

    out = model.predict_segmentation(
            inp=path,
            out_fname="/tmp/out.png"
        )

    plt.imshow(img),plt.show()
    plt.imshow(out),plt.show()

if __name__ == '__main__':  
    finetune_train()