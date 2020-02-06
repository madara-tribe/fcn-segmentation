import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from enum import IntEnum, Enum

ID1_aeroplane = (128, 0, 0)
ID2_bicycle = (0, 128, 0)

ID3_bird = (0,128,128)
ID4_boat = (128,0,0)
ID5_bottle = (128,0,128)
ID6_bus = (128,128,0)
ID7_car= (128,128,128)
ID8_cat= (0,0,64)
ID9_chair = (0,0,192)
ID10_cow= (0,128,64)
ID11_diningtable= (0,128,192)
ID12_dog = (128,0,64)
ID13_horse= (128,0,192)
ID14_motorbike= (128,128,64)
ID15_person = (128,128,192)
ID16_pottedplant = (0,64,0)
ID17_sheep= (0,64,128)
ID18_sofa = (0,192,0)
ID19_train = (0,192,128)
ID20_tvmonitor = (0,64,128)
ID21_void = (12,64,128)

class Class_ID(IntEnum):
    ID1_aeroplane = 1
    ID2_bicycle = 2
    ID3_bird = 3
    ID4_boat = 4
    ID5_bottle = 5
    ID6_bus = 6
    ID7_car= 7
    ID8_cat= 8
    ID9_chair = 9
    ID10_cow= 10
    ID11_diningtable= 11
    ID12_dog = 12
    ID13_horse= 13
    ID14_motorbike= 14
    ID15_person = 15
    ID16_pottedplant = 16
    ID17_sheep= 17
    ID18_sofa = 18
    ID19_train = 19
    ID20_tvmonitor = 20
    ID21_void = 21

    
def make_indexmap(img, base_indexmap):
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID1_aeroplane, Class_ID.ID1_aeroplane)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID2_bicycle, Class_ID.ID2_bicycle)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID3_bird, Class_ID.ID3_bird)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID4_boat, Class_ID.ID4_boat)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID5_bottle, Class_ID.ID5_bottle)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID6_bus, Class_ID.ID6_bus)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID7_car, Class_ID.ID7_car)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID8_cat, Class_ID.ID8_cat)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID9_chair, Class_ID.ID9_chair)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID10_cow, Class_ID.ID10_cow)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID11_diningtable, Class_ID.ID11_diningtable)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID12_dog, Class_ID.ID12_dog)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID13_horse, Class_ID.ID13_horse)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID14_motorbike, Class_ID.ID14_motorbike)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID15_person, Class_ID.ID15_person)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID16_pottedplant, Class_ID.ID16_pottedplant)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID17_sheep, Class_ID.ID17_sheep)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID18_sofa, Class_ID.ID18_sofa)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID19_train, Class_ID.ID19_train)
    base_indexmap = mask_to_indexmap(img, base_indexmap, ID20_tvmonitor, Class_ID.ID20_tvmonitor)
    indexmap = mask_to_indexmap(img, base_indexmap, ID21_void, Class_ID.ID21_void)
    
    return indexmap

def mask_to_indexmap(img, base_indexmap, class_color, class_id):
    h, w, _ = np.shape(img)
    masks = np.zeros([h, w, 3])  # (366, 500, 3)
    for x in range(h):
        for y in range(w):
            b, g, r = img[x, y]
            if (b, g, r) == class_color:
                masks[x, y] = class_id
    base_indexmap[masks == class_id] = class_id
    return base_indexmap



for idx, (anno_path, img_path) in enumerate(zip(sorted_anno_path[:20], sorted_img_path)):
    # annos
    img = cv2.imread(anno_path)
    print(img.shape, np.unique(img))
    h, w, _ = np.shape(img)
    img = cv2.resize(img, (int(w/4), int(h/4)), interpolation=cv2.INTER_NEAREST)
    plt.imshow(img),plt.show()
    nh, nw, _ = np.shape(img)
    base_indexmap = np.zeros([nh, nw, 3])


    indexmap = make_indexmap(img, base_indexmap)
    indexmap = cv2.resize(indexmap, (608, 416), interpolation=cv2.INTER_NEAREST)
    print(np.unique(indexmap), indexmap.shape)
    plt.imshow(indexmap),plt.show()


    save_dir="drive/My Drive/voc_anno"
    cv2.imwrite(os.path.join(save_dir, str(idx) + "indexmap.png"), indexmap)

    # imgs
    jpg = cv2.imread(img_path)
    jpg = cv2.resize(jpg, (608, 416), interpolation=cv2.INTER_NEAREST)
    save_dir2="drive/My Drive/vos_jpeg"
    cv2.imwrite(os.path.join(save_dir2, str(idx) + "indexmap.png"), jpg)
