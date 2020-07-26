from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import matplotlib.pyplot as plt
import tqdm
import sys

dataDir='COCOdataset2017'
dataType='val'
annFile='{}/annotations/instances_{}2017.json'.format(dataDir, dataType)

# Initialize the COCO api for instance annotations
coco=COCO(annFile)
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)
print(cats)


# In[13]:


def create_classes_list():
    lists=[]
    for i in range(len(cats)):
        lists.append(cats[i]['name'])
    return lists

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"
print('The class name is', getClassName(77, cats))


#classes = create_classes_list()
#print(classes)
def create_indexmap(img, anns, cats):
    mask = np.zeros((img['height'],img['width']))
    for k in range(0, len(anns)):
        className = getClassName(anns[k]['category_id'], cats)
        #print(className)
        pixel_value = filterClasses.index(className)+1
        mask = np.maximum(coco.annToMask(anns[k])*pixel_value, mask)
    return mask

    
H=608
W=416
jpg_save_dir='cocoseg/image'
anno_save_dir= 'cocoseg/ano'
# 11 classes
filterClasses= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'traffic light', 'stop sign', 'parking meter']
classes = filterClasses
error_files = []
for i in range(0, len(classes)):
    class_name = classes[i].split('\n')[0]
    catIds = coco.getCatIds(catNms=[class_name])
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of data of" + ' ' + class_name + ': ' + str(len(imgIds)))

    for j in range(0, len(imgIds)):
        img = coco.loadImgs(imgIds[j])[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        try:
            I = io.imread(img['coco_url'])
        except KeyboardInterrupt:
            sys.exit
        except:
            error_files.append(img['file_name'])
            continue
            
        mask = create_indexmap(img, anns, cats)
        
        #I = cv2.resize(I, (H, W))
        mask = mask.astype(np.uint8)
        cv2.imwrite(os.path.join(anno_save_dir, "class_{}_count_{}.png".format(i, j)), mask)
        
        cv2.imwrite(os.path.join(jpg_save_dir, "class_{}_count_{}.jpg".format(i, j)), I)
        #mask = cv2.resize(mask, (H, W), interpolation=cv2.INTER_NEAREST)
        #print(mask.shape, I.shape)
        #plt.imshow(mask),plt.show()
        #plt.imshow(I),plt.show()
        print(i,j, np.unique(mask))
