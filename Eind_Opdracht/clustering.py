from objdetection1 import *
import cv2 as cv
import numpy as np
from PIL import Image
import random
import math



def make_cluster(img):

    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    center[0] = (0,0,0)
    center[1] = (200,200,200)
    # center[2] = (100,0,0)
    # center[3] = (0,100,0)
    # center[0] = (0,0,100)

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    # cv.imshow('res2',res2)
    
    
    
    return res2


def createCandidatePositions(img):
    candidates = []
    bboxes = []
    # convert image to grayscale image
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # for i in range(10):
    #     cv.GaussianBlur(gray_image,(5,5),0)

    kernelClose = np.ones((50,50),np.uint8)
    kernelErode = np.ones((20,20),np.uint8)
    # kernelClose = np.ones((img.shape[1]//500,img.shape[0]//500),np.uint8)
    # kernelErode = np.ones((img.shape[1]//300,img.shape[1]//300),np.uint8)
    closing = cv.morphologyEx(gray_image, cv.MORPH_CLOSE, kernelClose)
    closing = cv.morphologyEx(closing, cv.MORPH_ERODE, kernelErode)
    closing = make_cluster(closing)

    edges = cv.Canny(closing,400,425,apertureSize = 3)
    # cv.imshow('res2',edges)

    # calculate moments of binary image
    # find contours in the binary image
    contours, hierarchy = cv.findContours(edges,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    
    
    # cv.drawContours(img, contours, -1, (255,0,0), 3)
    for c in contours:

        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        candidates.append(img[y:y+h,x:x+w])
        bboxes.append((x,y,(x+w),(y+h)))
        # calculate moments for each contour
        M = cv.moments(c)
        area = cv.contourArea(c)
    # cv.imshow('aids', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
   
    return np.array(candidates), np.array(bboxes)
