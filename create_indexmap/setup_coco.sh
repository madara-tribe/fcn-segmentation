#!/bin/sh
# COCO API
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make

# download 2017 train image
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# download 2017 vaildation image
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# download 2017 annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

