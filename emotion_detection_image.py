import os
import pandas as pd
import numpy as np
import cv2
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# from cf_matrix import make_confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model("saved_model")
labels = os.listdir("train")
imagePath = "Images/afiq.jpg"

img = cv2.imread(imagePath)
img = cv2.resize(img, (600,600))

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to gray scale

# initialize face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detect face using face classifier
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

def inference(img,face):
    # draw bbox
    for (x_coord, y_coord, width, height) in face:
        crop_img = img[y_coord:y_coord+height,x_coord:x_coord+width] # crop image to face only
        #cv2.rectangle(img, (x_coord, y_coord), (x_coord + width, y_coord + height), (0, 255, 0), 2)

    cv_image = cv2.cvtColor(np.array(crop_img), cv2.COLOR_BGR2GRAY) # change cropped image to grayscale

    resized = cv2.resize(cv_image, (48, 48), interpolation=cv2.INTER_AREA) # resize to 48x48

    probabilities = model.predict(np.array([resized]))
    emotion = labels[np.argmax(probabilities)]

    return emotion, probabilities


emotion_detection, confidence = inference(img, face)

result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for (x, y, w, h) in face:
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 4) #draw bounding box at face detected
    cv2.putText(result_img,emotion_detection, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA,False) # put text




plt.title("Emotion detection")
plt.imshow(result_img)
plt.show()