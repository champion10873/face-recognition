# Identify Two Images
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

def verify(img1_path, img2_path):
  img1 = cv2.imread(img1_path)
  img2 = cv2.imread(img2_path)
  plt.imshow(img1[:,:,::-1])
  plt.show()
  plt.imshow(img2[:,:,::-1])
  plt.show()
  output = DeepFace.verify(img1_path, img2_path)
  print(output)
  verification = output['verified']
  if verification:
    print('They are same')
  else:
    print('They are not same')

verify('img/imgA.jpg', 'img/imgB.jpg')