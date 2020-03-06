import cv2
import sys
import os
paths = os.path.abspath(os.path.dirname(__file__))
img1 = os.path.join(paths,'img/standard.png')
print(img1)
img = cv2.imread(img1)
face_path = os.path.join(paths,'img/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_path)
def get_face(img2):
        x,y,w,h = 91,23,40,40 #standard face location
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,5)
        if len(faces)==0:
            return None,None,None
        (x1,y1,w1,h1) = faces[0]
        
        imgs = img2[y1+int(h1*0.2):y1+h1,x1+int(w1*0.1):x1+w1-int(w1*0.1)]
        imgs = cv2.resize(imgs,(w,h+10)) # input img
        img[y-10:y+h,x:x+w] = imgs # overlap raw img
        boxs =[]
        boxs.append([x,y-10,h,w])
        boxs.append([x1+int(w1*0.1),y1+int(h1*0.2),y1+h1,x1+w1-int(w1*0.1)])
        return img,boxs,img2
