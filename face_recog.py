import cv2
import numpy as np
import face_recognition

img_erwin = face_recognition.load_image_file('img/imgA.jpg')
img_erwin = cv2.cvtColor(img_erwin, cv2.COLOR_BGR2RGB)

# To find the face location
face = face_recognition.face_locations(img_erwin)[0]

# Converting image into encodings
train_encode = face_recognition.face_encodings(img_erwin)[0]

# Lets test an image
test = face_recognition.load_image_file('img/imgB.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
test_encode = face_recognition.face_encodings(test)[0]

print(face_recognition.compare_faces([train_encode], test_encode))
cv2.rectangle(img_erwin, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 1)
cv2.imshow('img_erwin', img_erwin)
cv2.waitKey(0)