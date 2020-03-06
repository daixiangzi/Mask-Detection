# Mask-Detection
we don't use any face maks dataset to complete face mask Detection, we think mask material is similar to clothe,they are textiles.  
# Dataset  
---
Clothing Co-Parsing(CCP) dataset that i used can be found https://github.com/bearpaw/clothing-co-parsing.

Pre-Processing
---
After importing the image data , divide into four classes(Background, hair, skin, clothes) and resize 384 x 256.

Models
---
Models are found in model file. Model file includes FCN.py(Fully Covolutional Networks), DeepLabV2.py(DeepLab V2), UNet.py(U-Net).
U-Net has shown the best performance among the models in this project.


Requirement
---
* Python
* Keras
* Python packages : numpy, matplotlib, opencv, and so on...


