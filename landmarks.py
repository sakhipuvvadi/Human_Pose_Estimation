# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:04:50 2025

@author: sakhi
"""

import mediapipe as mp
import cv2

#intialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# load an image
img_path = "pose1.jpg"
img = cv2.imread(img_path,1)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# perform pose estimation
results = pose.process(img_rgb)

# draw landmarks only (no lines)
if results.pose_landmarks:
    print("Pose landmarks detected!")
    
    for idx,landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x:{landmark.x}, y:{landmark.y}, z:{landmark.z}, visibility:{landmark.visibility})")
    for landmark in results.pose_landmarks.landmark:
        h,w,c = img.shape
        
        #convert normalized coordinates to pixel coordinates
        cx,cy = int(landmark.x*w),int(landmark.y*h)
        
        #draw key points
        cv2.circle(img, (cx,cy), 5, (0,255,0),-1)
        
        #optional: Draw landmarks on the image
        annotated_img = img.copy()
        mp_drawing.draw_landmarks(annotated_img, results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
cv2.imshow("pose landmarks",img)
cv2.imshow("annotated img", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#close all the resources
pose.close()
