# fcn-segmentation performance

<b>voc image1</b><hr>

<b>image</b>
![image1](https://user-images.githubusercontent.com/48679574/74220351-b7f05900-4cf2-11ea-996e-f13c5a0252e1.png)



![voc_image1](https://user-images.githubusercontent.com/48679574/73957794-40b17280-494a-11ea-845f-734f4fa94c86.png)

<b>result</b>

![voc_result1](https://user-images.githubusercontent.com/48679574/73957815-4b6c0780-494a-11ea-8179-87460af9e61b.png)



<b>voc image2</b><hr>

<b>image</b>

![voc_image2](https://user-images.githubusercontent.com/48679574/73957967-84a47780-494a-11ea-849d-af3b5aebad7b.png)

<b>result</b>

![voc_result2](https://user-images.githubusercontent.com/48679574/73957978-89692b80-494a-11ea-9d4d-c793d24c3de1.png)


<b>original image</b><hr>

<b>image</b>

![original_img1](https://user-images.githubusercontent.com/48679574/73958093-ba496080-494a-11ea-9d81-4dcaa2a2c2dc.png)

<b>result</b>

![original_result1](https://user-images.githubusercontent.com/48679574/73958109-bfa6ab00-494a-11ea-9fc6-9ebada69ce3e.png)

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
