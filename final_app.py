# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:53:49 2025

@author: sakhi
"""

from PIL import Image
import cv2
import streamlit as st
import numpy as np
from io import BytesIO

demo_img = "run.jpg"

body_parts = {"Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,
              "LElbow":6,"LWrist":7,"RHip":8,"RKnee":9,"RAnkle":10,"LHip":11,"LKnee":12,
              "LAnkle":13,"REye":14,"LEye":15,"REar":16,"LEar":17,"Background":18}

pose_pairs = [["Neck","RShoulder"],["Neck","LShoulder"],["RShoulder","RElbow"],["RElbow","RWrist"]
              ,["LShoulder","LElbow"],["LElbow","LWrist"],["Neck","RHip"],["RHip","RKnee"],
              ["RKnee","RAnkle"],["Neck","LHip"],["LHip","LKnee"],["LKnee","LAnkle"],["Neck","Nose"]
              ,["Nose","REye"],["REye","REar",],["Nose","LEye"],["LEye","LEar"]]

width=370
height=370
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

st.title("Human Pose Estimation Using Machine Learning")
st.text("Make sure the image is clear and all parts are clearly visible")

img_file = st.file_uploader("Upload a clear image",type=["png","jpg","jpeg"])

if img_file is not None:
    image = np.array(Image.open(img_file))
else:
    image = np.array(Image.open(demo_img))

st.subheader("Original Image")
st.image(image,caption=f"Original Image",use_column_width=True)
thres = st.slider("Threshold for detecting key points",min_value=0,value=20,max_value=100,step=5)
thres = thres/100

@st.cache
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2RGB)
    
    net.setInput(cv2.dnn.blobFromImage(frame,1.0,(inWidth,inHeight),(127.5,127.5,127.5),swapRB=True,crop=False))
    
    out = net.forward()
    out = out[:,:19,:,:]
    
    assert(len(body_parts)==out.shape[1])
    
    points=[]
    
    for i in range(len(body_parts)):
        heatmap = out[0,i,:,:]
        _,conf,_,point = cv2.minMaxLoc(heatmap)
        x = (frameWidth * point[0])/out.shape[3]
        y = (frameHeight * point[1])/out.shape[2]
        points.append((int(x),int(y)) if conf > thres else None)
        
    for pair in pose_pairs:
        partfrom = pair[0]
        partto = pair[1]
        assert(partfrom in body_parts)
        assert(partto in body_parts)
        
        idfrom = body_parts[partfrom]
        idto = body_parts[partto]
        
        if points[idfrom] and points[idto]:
            cv2.line(frame,points[idfrom],points[idto],(0,255,0),3)
            cv2.ellipse(frame, points[idfrom],(3,3),0,0,360,cv2.FILLED )
            cv2.ellipse(frame, points[idto],(3,3),0,0,360,cv2.FILLED)
    
    t,_ = net.getPerfProfile()
    return frame

output = poseDetector(image)
st.subheader("Pose Estimated")
st.image(output,caption=f"Position Estimated",use_column_width=True)
output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
buf = BytesIO()
output_pil.save(buf, format="PNG")
byte_im = buf.getvalue()
st.download_button(
    label="Download Pose-Estimated Image",
    data=byte_im,
    file_name="pose_estimation_output.png",
    mime="image/png",
)
        
    
    