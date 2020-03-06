# Mask-Detection
We don't use any face maks dataset to complete face mask Detection, we think mask material is similar to clothes,they are textiles.  
Finally we count clothes class of pixel proportion of face to predict by segmentation network.  
# Dataset  
---
Clothing Co-Parsing(CCP) dataset that i used can be found https://github.com/bearpaw/clothing-co-parsing.

Pre-Processing
---
After importing the image data , divide into four classes(Background, hair, skin, clothes) and resize 384 x 256.

Models
---
Models are found in model file. Model file includes FCN.py DeepLabV2.py, UNet.py.
U-Net has shown the best performance among the models in this project.


Requirement
---
* Python3
* Keras
# Usage 
Then copy the annotation and photo files into the same path as pre_processing.py in the download file.
After that you copy files, you can run train model.
## train model and test image  
```python
python3 train.py
```
After,you can test your image  
```python
python3 test.py --model_path=/home/***.h5 --image_path=/home/yout_test_img_path.jpg  
```
[Pre_model link](https://pan.baidu.com/s/1Lrkzs7rgPBTbDAvgmWj5Bw) `Extraction code`:piwi  

# Result  
![Mask](https://github.com/daixiangzi/Mask-Detection/tree/master/img/mask.jpg) ![No_mask](https://github.com/daixiangzi/Mask-Detection/tree/master/img/n0_mask.jpg)




