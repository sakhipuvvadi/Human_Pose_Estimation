# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:14:44 2025

@author: sakhi
"""

import cv2
img = cv2.imread(r"foot.png",1)
print(img.shape)
height = 200
width = 150
dim = (width,height)
resized = cv2.resize(img, dim)
cv2.imshow("original.png", img)
cv2.imshow("resized.png",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()