import cv2
import sys
import os
import shutil
import xml.etree.ElementTree import *

image_base = "dataset/helen/"
cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml"

tree = parse('test.xml')
root = tree.getroot()
images = root.find('images')


for line in open('dataset/helen/testname.txt', 'r'):
    line = line.strip()
    image = cv2.imread(image_base + line + '.jpg', 0)
    if image is None:
        print('Not open: ', line)
        quit()
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(
        image, scaleFactor=1.2, minNeighbors=2, minSize=(1, 1))

    for rect in facerect:
        _x, _y, _width, _height = rect

    dst_img = image[y:y+height, x:x+width]
    face_img = cv2.resize(dst_img, (100, 100))
    cv2.imwrite('dataset/helen_face/' + line + '.jpg', face_img)
