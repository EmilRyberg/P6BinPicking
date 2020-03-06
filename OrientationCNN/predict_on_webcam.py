# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:19:00 2019

@author: Emil
"""

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import argparse


def main(model_name):
    model = tf.keras.models.load_model(model_name)
    video = cv2.VideoCapture(6)
    while True:
        _, frame = video.read()
        
        img = Image.fromarray(frame, 'RGB')

        img = img.resize((224, 224))
        img_array = np.array(img)
        #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_array, axis=0) / 255

        #Calling the predict method on model to predict 'me' on the image
        predictions = model.predict(img_array)
        text_str = "Is facing right: {0:.2f}%".format(predictions[0, 0] * 100.0)
        cv2.putText(frame, text_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 0), 2, cv2.LINE_AA)

        cv2.imshow("Capturing", frame)
        #cv2.imshow("Resized", np.squeeze(img_array, axis=0))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('orientation_cnn.hdf5')