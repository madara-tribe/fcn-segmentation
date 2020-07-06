from google.colab import drive
drive.mount('/content/drive')


import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pascal_voc_classID import Class_ID
import glob

class Indexmaps:
    def __init__(self):
        self.ID1 = (0, 0, 255)
        self.ID2 = (255, 255, 0)
        self.ID3 = (255, 0, 0)
        self.ID4 = (69, 47, 142)
        self.ID5 = (193, 214, 0)
        self.ID6 = (180, 0, 129)
        self.ID7 = (65, 166, 1)
        self.ID8 = (255, 134, 0)
        self.ID9 = (136, 45, 66)
        self.ID10 = (0, 255, 255)

    
    def make_indexmap(self, img, base_indexmap):
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID1, Class_ID.ID1_car)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID2, Class_ID.ID2_signal)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID3, Class_ID.ID3_pedestrian)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID4, Class_ID.ID4_lane)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID5, Class_ID.ID5_bus)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID6, Class_ID.ID6_truck)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID7, Class_ID.ID7_motorbike)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID8, Class_ID.ID8_signs)
        base_indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID9, Class_ID.ID9_ground)
        indexmap = self.mask_to_indexmap(img, base_indexmap, self.ID10, Class_ID.ID10_sidewalk)
        
        return indexmap

    def mask_to_indexmap(self, img, base_indexmap, class_color, class_id):
        h, w, _ = np.shape(img)
        masks = np.zeros([h, w, 3])  # (366, 500, 3)
        for x in range(h):
            for y in range(w):
                b, g, r = img[x, y]
                if (b, g, r) == class_color:
                    masks[x, y] = class_id
        base_indexmap[masks == class_id] = class_id
        return base_indexmap

def create_resized_img(img, H=608, W=416):
      resize_img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
      #plt.imshow(resize_img),plt.show()
      h, w, _ = np.shape(resize_img)
      base_indexmap = np.zeros([h, w, 3])
      return resize_img, base_indexmap

def load_img(path):
      img = cv2.imread(path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      return img


image_dir = '/content/drive/My Drive/seg_train_images'
anno_dir = '/content/drive/My Drive/seg_train_annotations'
if not os.path.exists(image_dir) or not os.path.exists(anno_dir):
    print("ERROR!The folder is not exist")
    
sorted_img_path = [os.path.join(image_dir, path) for path in sorted(glob.glob(image_dir+'/*.jpg'))]
sorted_anno_path = [os.path.join(anno_dir, path) for path in sorted(glob.glob(anno_dir+'/*.png'))]
print(len(sorted_img_path), len(sorted_anno_path))     

env = Indexmaps()
anno_save_dir="/content/drive/My Drive/indexmap"
jpg_save_dir = "/content/drive/My Drive/resized_train"

for idx, (anno_path, jpg_path) in enumerate(zip(sorted_anno_path, sorted_img_path)):
    # annos
    save_name, _ = os.path.splitext(os.path.basename(anno_path))
    img = load_img(anno_path)

    # create indexmap
    resize_img, base_indexmap = create_resized_img(img, H=608, W=416)

    print(resize_img.shape, base_indexmap.shape)
    indexmap = env.make_indexmap(resize_img, base_indexmap)
    print(np.unique(indexmap), indexmap.shape)
    plt.imshow(indexmap),plt.show()

    # save anno
    cv2.imwrite(os.path.join(anno_save_dir, save_name + ".png"), indexmap)

    # save jpg
    print(jpg_path)
    jpg = load_img(jpg_path)
    resized_jpg, _ = create_resized_img(jpg, H=608, W=416)
    plt.imshow(resized_jpg),plt.show()
    cv2.imwrite(os.path.join(jpg_save_dir, save_name + ".jpg"), resized_jpg)

    if idx==50:
      break



