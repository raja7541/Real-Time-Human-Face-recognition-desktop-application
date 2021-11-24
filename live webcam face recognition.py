# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:30:45 2021

@author: ELCOT
"""

import cv2

trainedhumanface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facewebcam = cv2.VideoCapture(0)

while True:
    success,frame=facewebcam.read()
    if success==True:
        gray_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=trainedhumanface.detectMultiScale(gray_image)
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),6)
        cv2.imshow('facewebcam',frame)
        key=cv2.waitKey(1)
        if key==ord("e"):
            break

facewebcam.release()
cv2.destroyAllWindows()