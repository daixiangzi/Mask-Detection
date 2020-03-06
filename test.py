import numpy as np
from data_utils import *
import cv2
import matplotlib.pyplot as plt
import matplotlib
import argparse
from keras.models import load_model
import time
import tensorflow as tf
from detec_face import get_face
from collections import  Counter
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
"""
for example:
python3 ***.py --model_path=your_model_path --image_path=yout_test_img_path
"""
# get arguments
def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', dest = 'model_path',default = './save_model/UNet_model.h5', help ='load model path')
    parser.add_argument('--image_path', dest = 'image_path', help ='image path')

    return parser.parse_args()

# main function
def main(args):
    # data load
    IMG_HEIGHT = 384
    IMG_WIDTH = 256
    #cap =cv2.VideoCapture(0)
    img = cv2.imread(args.image_path, 1)
    model = load_model(args.model_path)
    #while 1:
        #ret,img = cap.read()
        #assert ret==True,"cap failed"
    img,boxs,face_img = get_face(img)
    print(boxs)
    assert boxs!=None,"no face"
    face_img = np.array(face_img)
    img = np.array(img)
    areas = boxs[0][2]*boxs[0][3]
        #predict
    start_time = time.time()
    test_img = model.predict(img.reshape([1, 384, 256, 3])/255.)
    end_time = time.time()-start_time
    print("use time:%s"%(str(end_time)))
    test_img = np.array(test_img)
    


    argm = np.argmax(test_img, axis=3)
    temp = np.squeeze(argm)
    counts = temp[boxs[0][1]:boxs[0][1]+boxs[0][2],boxs[0][0]:boxs[0][0]+boxs[0][3]]
    pre = np.sum(counts==3)/float(areas) #skin pixel Statistics
    print(pre)
    if pre>=0.3:
         print("Mask")
         cv2.rectangle(face_img, (boxs[1][0],boxs[1][1]), (boxs[1][3],boxs[1][2]), (0, 255, 0), 2)
    else:
         print("No Mask")
         cv2.rectangle(face_img, (boxs[1][0],boxs[1][1]), (boxs[1][3],boxs[1][2]), (255, 0, 0), 2)
    plt.cla()
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(face_img)
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(counts)
    plt.pause(0.1)
    ax1.cla()
    ax2.cla()
    #matplotlib.image.imsave("filename.png",temp)


if __name__ == '__main__':

    args = arg_parser()
    print("Args : ", args)
    main(args)
