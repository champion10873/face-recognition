import cv2
import numpy as np
import face_recognition
img_modi = face_recognition.load_image_file('img/imgC.jpg')
img_modi_rgb = cv2.cvtColor(img_modi,cv2.COLOR_BGR2RGB)
# ------------ Detecting Face ----------------
face = face_recognition.face_locations(img_modi_rgb)[0]
copy = img_modi_rgb.copy()
# ------------ Drawing Rectangle ----------------
cv2.rectangle(copy, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)
cv2.imshow('copy', copy)
cv2.imshow('MODI', img_modi_rgb)
cv2.waitKey(0)