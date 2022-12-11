
import cv2
import streamlit as st
import numpy as np 
from PIL import Image


def cartoonization (img, cartoon):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    if cartoon == "Pencil Sketch":
       h, w, c = img.shape
       img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
       img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
       img_blend = cv2.divide(img_gray, img_blur, scale=256)
       img_blend = cv2.resize(img, (w, h))
       img_blend = cv2.multiply(img_blend, img, scale=1. / 256)
       img_blend = cv2.adaptiveThreshold(img_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  blockSize=9,
                                  C=2)
       cartoon= cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)

    if cartoon == "Cartoon-1":
        h, w, c = img.shape
        img_color = cv2.resize(img, (w, h))
        img_color = cv2.pyrDown(img_color)
        img_color = cv2.bilateralFilter(img_color, 9, 9,7)
        img_color = cv2.pyrUp(img_color)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_color = cv2.resize(img, (w, h))
        cartoon = cv2.bitwise_and(img_color, img_edge)


    if cartoon == "Cartoon-2": 
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, 9)
        edges = cv2.adaptiveThreshold(gray_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
        data = np.float32(img).reshape((-1,3))
        criteria = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        ret, label, center = cv2.kmeans(data, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result=center[label.flatten()]
        result = result.reshape(img.shape)
        blurred = cv2.bilateralFilter(result, 4,200,200)
        cartoon = cv2.bitwise_and(blurred, blurred)

    return cartoon

###############################################################################
    
st.write("""
          # Cartoonify

          """
          )

st.write("This is an app to cartoonize your photos")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])

if file is None:
    st.text("File Not Uploaded")
else:
    image = Image.open(file)
    img = np.array(image)
    
    option = st.sidebar.selectbox(
    'Choose a cartoon Filter!!!',
    ('Pencil Sketch', 'Cartoon-1', 'Cartoon-2'))
    
    st.text("Original image")
    st.image(image, use_column_width=True)
    
    st.text("Cartoonized image")
    cartoon = cartoonization(img, option)
    
    st.image(cartoon, use_column_width=True)

