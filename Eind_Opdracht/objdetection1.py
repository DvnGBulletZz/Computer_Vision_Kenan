import pandas as pd
import numpy as np
import cv2 as cv

import tensorflow as tf
import os
from PIL import Image

# from keras import layers as L
# from keras.models import Model
# from keras.applications import MobileNetV2




def loadData(dir):
    df = pd.read_csv(dir+'/_annotations.csv')
    file = df['filename'].to_numpy()
    width = df['width'].to_numpy()
    height = df['height'].to_numpy()
    label = df['class'].to_numpy()
    xMin = df['xmin'].to_numpy()
    xMax = df['xmax'].to_numpy()
    yMin = df['ymin'].to_numpy()
    yMax = df['ymax'].to_numpy()

    return file,width, height,label, xMin, xMax, yMin, yMax

def cut_bbox(i, dir):
    file,width, height,label, xMin, xMax, yMin, yMax = loadData(dir)
    file,width ,height, label, xMin, xMax, yMin, yMax = file[i], width ,height ,label[i], xMin[i], xMax[i], yMin[i], yMax[i], 
    directory = dir+'/'+str(file)
    print(directory)
    
    img = cv.imread(directory)
    print(yMin)
    print(yMax)
    print(xMin)
    print(xMax)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    

    cropped_img = img[int(yMin):int(yMax), int(xMin):int(xMax)]
    return np.array(cropped_img)




def change_bboximg( img, size ):
    # print(img)
    fill_bboximg = img_border( resize_img( img, size ), size )
    borderColor = (125, 125, 125)
    if( fill_bboximg.shape[0] < size[1] ):
        fill_bboximg = cv.copyMakeBorder(fill_bboximg,0,1,0,0,cv.BORDER_CONSTANT,value=borderColor)
    if( fill_bboximg.shape[1] < size[0] ):
        fill_bboximg = cv.copyMakeBorder(fill_bboximg,0,0,0,1,cv.BORDER_CONSTANT,value=borderColor)
    return np.array(fill_bboximg)

def resize_img(img, size):
    resized_img = img
    if resized_img.shape[0] > size[1]:
        scale = size[1] / resized_img.shape[0]
        resized_img = cv.resize(resized_img, (int(resized_img.shape[1] * scale), size[1]))
    if resized_img.shape[1] > size[0]:
        scale = size[0] / resized_img.shape[1]
        resized_img = cv.resize(resized_img, (size[0], int(resized_img.shape[0] * scale)))
        # print(resize_img)
    return np.array(resized_img)


def img_border(img , size):
    border_img = img
    color = (125,125,125)
    
    if border_img.shape[0] < size[1]:
        border_img = cv.copyMakeBorder(border_img,0,1,0,0,cv.BORDER_CONSTANT,value=color)
    if border_img.shape[1] < size[0]:
        border_img = cv.copyMakeBorder(border_img,0,0,0,1,cv.BORDER_CONSTANT,value=color)
    return np.array(border_img)

def import_dataset(dirs):
    new_data , labels= [], []
    for dir in dirs:
        for j in range(len(loadData(dir)[0])):
            new_data.append(change_bboximg(cut_bbox(j, dir), (224,224)))
            if( loadData(dir)[3][j] == "xyzal 5mg" ):
                labels.append(0)
            elif( loadData(dir)[3][j] == "Cipro" ):
                labels.append(1)
            elif( loadData(dir)[3][j] == "Ibuphil Cold 400-60" ):
                labels.append(2)
            elif( loadData(dir)[3][j] == "red" ):
                labels.append(3)
            elif( loadData(dir)[3][j] == "pink" ):
                labels.append(4)
            elif( loadData(dir)[3][j] == "white" ):
                labels.append(5)
            elif( loadData(dir)[3][j] == "blue" ):
                labels.append(6)
            saveCroppedImage( change_bboximg(cut_bbox(j), (224,224)), 'cropped'+ loadData(dir)[0][j]) # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

    return np.array(new_data), np.array(labels)

def saveCroppedImage( croppedImage, filePath, imageName ):
    im = Image.fromarray(croppedImage)
    im.save(filePath + '/' + imageName)



flowers_labels_dict = {
    'xyzal 5mg': 0,
    'Cipro': 1,
    'Ibuphil Cold 400-60': 2,
    'red': 3,
    'pink': 4,
    'white': 5,
    'blue': 6,
    
}
dirs = ["train", "test", "valid"]


import_dataset(dirs)
# for i in range(len(loadData()[0])):
    

