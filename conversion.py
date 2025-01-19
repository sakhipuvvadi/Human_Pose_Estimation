# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:59:11 2025

@author: sakhi
"""

import cv2
img = cv2.imread(r"foot.png",1)
print(img)
img4 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img5 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img6 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img7 = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow("color.png", img)
cv2.imshow("gray.png", img4)
cv2.imshow("rgb.png", img5)
cv2.imshow("hsv.png", img6)
cv2.imshow("lab.png", img7)
cv2.waitKey(0)
cv2.destroyAllWindows()