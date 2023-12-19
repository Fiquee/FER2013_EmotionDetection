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

video_capture = cv2.VideoCapture(0) # 0 for camera, video path for video file


# initialize face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def inference(img,face):
    # draw bbox
    for (x_coord, y_coord, width, height) in face:
        crop_img = img[y_coord:y_coord+height,x_coord:x_coord+width]
        #cv2.rectangle(img, (x_coord, y_coord), (x_coord + width, y_coord + height), (0, 255, 0), 2)

    cv_image = cv2.cvtColor(np.array(crop_img), cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(cv_image, (48, 48), interpolation=cv2.INTER_AREA)
    probabilities = model.predict(np.array([resized]))
    emotion = labels[np.argmax(probabilities)]

    return emotion, probabilities

def plot_bounding_box(img,face,emotion_detection):
    # result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(img,emotion_detection, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA,False)

    return img


while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    # print(faces)

    if(len(faces) > 0):
        emotion, confidence = inference(video_frame,faces)

        video_frame = plot_bounding_box(video_frame,faces,emotion)

    
    cv2.imshow("Emotion Detection", video_frame)  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()