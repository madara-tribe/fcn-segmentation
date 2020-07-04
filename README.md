# fcn-segmentation performance

<b>voc image1</b><hr>

<b>image</b> and <b>result</b>

<img src="https://user-images.githubusercontent.com/48679574/73957794-40b17280-494a-11ea-845f-734f4fa94c86.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/73957815-4b6c0780-494a-11ea-8179-87460af9e61b.png" width="400px">





<b>voc image2</b><hr>

<b>image</b> and <b>result</b>

<img src="https://user-images.githubusercontent.com/48679574/73957967-84a47780-494a-11ea-849d-af3b5aebad7b.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/73957978-89692b80-494a-11ea-9d4d-c793d24c3de1.png" width="400px">



<b>original image</b><hr>

<b>image</b> and <b>result</b>

<img src="https://user-images.githubusercontent.com/48679574/73958093-ba496080-494a-11ea-9d81-4dcaa2a2c2dc.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/73958109-bfa6ab00-494a-11ea-9fc6-9ebada69ce3e.png" width="400px">



# indexmap pixel value range

<b>annotation png image pixel</b>

In the case of pascal_voc image taht has 21 classes, its pixel ranges from 0 to 21
```python
import numpy as np
anno = cv2.imreead(annos_path)
np.unique(anno)

>>>
[0,1,2,3,4,5,6,7,8,.....,21]

```

To create indexmap, look ```create_indexmap/pascal_voc_indexmap.py```

# train and predict code


```
num_cls = 51
input_height = 416
input_width = 608


model = fcn_32(n_classes = num_cls, input_height=input_height, input_width=input_width)
directory = 'tmp'
os.makedirs(directory, exist_ok=True)
model.train(
        train_images =  "dataset/images_train",
        train_annotations = "dataset/annotations_train",
        val_images="dataset/images_test",
        val_annotations="dataset/annotations_test",
        checkpoints_path = directory, epochs=30
    )
    
    
out = model.predict_segmentation(
    inp="dataset/N.png",
    out_fname="/tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model 
print(model.evaluate_segmentation(inp_images_dir="dataset/images_test/"  , annotations_dir="dataset/annotations_test/" ) )
```

# references

- [keras-segmentation](https://github.com/divamgupta/image-segmentation-keras)
- [prepare dataset](https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view)

