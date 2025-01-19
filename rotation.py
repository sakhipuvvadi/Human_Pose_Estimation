# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:26:36 2025

@author: sakhi
"""

import cv2

img = cv2.imread(r"logo.png",1)
h,w = img.shape[:2]
center = (w/2,h/2)

M = cv2.getRotationMatrix2D(center, 90, 1)
rotated90 = cv2.warpAffine(img, M, (h,w))

M = cv2.getRotationMatrix2D(center, 120, 1)
rotated180 = cv2.warpAffine(img, M, (w,h)

M = cv2.getRotationMatrix2D(center, 270, 1)
rotated270 = cv2.warpAffine(img, M, (h,w))

cv2.imshow("0.png", img)
cv2.imshow("90.png", rotated90)
cv2.imshow("180.png",rotated180)
cv2.imshow("270.png",rotated270)

cv2.waitKey(0)
cv2.destroyAllWindows()