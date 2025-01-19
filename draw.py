# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:58:26 2025

@author: sakhi
"""

import cv2
img = cv2.imread(r"people.png",1)
cv2.circle(img, (577,150), 90, (0,0,255),4)
cv2.imshow("circle.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(r"people.png",1)
cv2.rectangle(img, (313,128),(520,290), (0,255,0),2)
cv2.imshow("rectangle.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(r"people.png",1)
cv2.ellipse(img,(186,219),(150,100),0,0,360,(255,0,0),3)
cv2.imshow("ellipse.png",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(r"people.png", 1)
cv2.line(img, (823,25), (25,554), (0,0,0),5)
cv2.imshow("line.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()