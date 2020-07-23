from google.colab import drive
drive.mount('/content/drive')


import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pascal_voc_classID import BGR
import glob

class Indexmaps:
    def __init__(self):
        self.ID1 = (0, 0, 255)
        self.ID2 = (142, 47, 69)
        self.ID3 = (255, 0, 0)
        self.ID4 = (0, 255, 255)
        
        
       # self.ID5 = (0, 214, 193)
       # self.ID6 = (129, 0, 180)
        #self.ID7 = (1, 166, 65)
        #self.ID8 = (0, 134, 255)
        #self.ID9 = (66, 45, 136)
        #self.ID10 = (255, 255, 0)

    
    def make_indexmap(self, img, base_indexmap):
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID1, BGR.ID1_pedestrian)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID2, BGR.ID2_lane)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID3, BGR.ID3_car)
        indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID4, BGR.ID4_signal)
        #base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID4, BGR.ID4_lane)
        #base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID5, BGR.ID5_bus)
        #base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID6, BGR.ID6_truck)
        #base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID7, BGR.ID7_motorbike)
       # base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID8, BGR.ID8_signs)
        #base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID9, BGR.ID9_ground)
        
        
        return indexmap

    def mask_to_indexmap(self, img, base_indexmap, class_color, class_id):
        h, w, _ = np.shape(img)
        masks = np.zeros([h, w, 3])  # (366, 500, 3)
        for x in range(h):
            for y in range(w):
                b, g, r = img[x, y]
                if (b, g, r) == class_color:
                    masks[x, y] = class_id
                else:
                    continue
        base_indexmap[masks == class_id] = class_id
        return base_indexmap

def create_resized_img(img, H=608, W=416):
      resize_img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
      #plt.imshow(resize_img),plt.show()
      h, w, _ = np.shape(resize_img)
      base_indexmap = np.zeros([h, w, 3])
      return resize_img, base_indexmap


image_dir = '/content/drive/My Drive/seg_train_images'
anno_dir = '/content/drive/My Drive/seg_train_annotations'
if not os.path.exists(image_dir) or not os.path.exists(anno_dir):
    print("ERROR!The folder is not exist")
    
sorted_img_path = [os.path.join(image_dir, path) for path in sorted(glob.glob(image_dir+'/*.jpg'))]
sorted_anno_path = [os.path.join(anno_dir, path) for path in sorted(glob.glob(anno_dir+'/*.png'))]
print(len(sorted_img_path), len(sorted_anno_path))     

env = Indexmaps()
anno_save_dir="/content/drive/My Drive/indexmap"
jpg_save_dir = "/content/drive/My Drive/resize_train"

for idx, (anno_path, jpg_path) in enumerate(zip(sorted_anno_path, sorted_img_path)):
    # annos
    save_name, _ = os.path.splitext(os.path.basename(anno_path))
    img = cv2.imread(anno_path)

    # create indexmap
    resize_img, base_indexmap = create_resized_img(img, H=608, W=416)

    print(resize_img.shape, base_indexmap.shape)
    indexmap = env.make_indexmap(resize_img, base_indexmap)
    print(np.unique(indexmap), indexmap.shape)
    plt.imshow(indexmap),plt.show()

    # save anno
    indexmap = indexmap.astype(np.uint8)
    cv2.imwrite(os.path.join(anno_save_dir, save_name + ".png"), indexmap)

    # save jpg
    print(jpg_path)
    jpg = cv2.imread(jpg_path)
    resized_jpg = cv2.resize(jpg, (608, 416))
    plt.imshow(resized_jpg),plt.show()
    cv2.imwrite(os.path.join(jpg_save_dir, save_name + ".jpg"), resized_jpg)

    if idx==30:
      break