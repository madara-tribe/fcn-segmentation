#!/bin/sh
# COCO API
pip3 install Cython
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
CUDA_VISIBLE_DEVICES=1 python 〜.py

# download 2017 train image
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip

# download 2017 vaildation image
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# download 2017 annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

