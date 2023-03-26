import pandas as pd
import numpy as np
import cv2 as cv

import tensorflow as tf
import os
from PIL import Image
import random

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
    # print(directory)
    
    img = cv.imread(directory)
    # print(yMin)
    # print(yMax)
    # print(xMin)
    # print(xMax)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    

    cropped_img = img[int(yMin):int(yMax), int(xMin):int(xMax)]
    return np.array(cropped_img)




def change_bboximg( img, size ):
    # print(img)
    fill_bboximg = img_border( resize_img( img, size ), size )
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
        Y = int((size[1]-border_img.shape[0])/2)
        border_img = cv.copyMakeBorder(border_img,Y,Y,0,0,cv.BORDER_CONSTANT,value=color)
    if border_img.shape[1] < size[0]:
        X= int((size[0]-border_img.shape[1])/2)
        border_img = cv.copyMakeBorder(border_img,0,0,X,X,cv.BORDER_CONSTANT,value=color)
    if( border_img.shape[0] < size[1] ):
        border_img = cv.copyMakeBorder(border_img,0,1,0,0,cv.BORDER_CONSTANT,value=color)
    if( border_img.shape[1] < size[0] ):
        border_img = cv.copyMakeBorder(border_img,0,0,0,1,cv.BORDER_CONSTANT,value=color)
    return np.array(border_img)

def intersect_bbox( bboxX, bboxY, randomX, randomY):
    for y in range(randomY[0], randomY[1]):
        for x in range( randomX[0], randomX[1] ):
            if( (x == bboxX[0] and y == bboxY[0]) or (x == bboxX[1] and y == bboxY[1]) ):
              return True
    return False

def cut_random( img, size, XminBB, XmaxBB , YminBB, YmaxBB ):

    Xmin = random.randint(0, img.shape[1]-size[0])
    Ymin = random.randint(0, img.shape[0]-size[1])
    Xmax = Xmin + size[0]
    Ymax = Ymin + size[1]
   

    while intersect_bbox( (XminBB, XmaxBB), (YminBB, YmaxBB), (Xmin, Xmax), (Ymin, Ymax)):
        Xmin = random.randint(0, img.shape[1]-size[0])
        Ymin = random.randint(0, img.shape[0]-size[1])
        Xmax = Xmin + size[0]
        Ymax = Ymin + size[1]
        

    newImage = img[Ymin:Ymax, Xmin:Xmax]
    return np.array(newImage)

def import_dataset(dirs):
    new_data , labels= [], []
    
    for dir in dirs:
        file,width, height,label, xMin, xMax, yMin, yMax = loadData(dir)
        for j in range(len(loadData(dir)[0])):
            new_data.append(change_bboximg(cut_bbox(j, dir), (224,224)))
            # print(loadData(dir)[3][j])
            # if( loadData(dir)[3][j] == "Xyzall 5mg" ):
            #     labels.append(1)
            labels.append(pill_name_dict[loadData(dir)[3][j]])
            # elif( loadData(dir)[3][j] == "Cipro 500" ):
            #     labels.append(1)
            # elif( loadData(dir)[3][j] == "Ibuphil Cold 400-60" ):
            #     labels.append(1)
            # elif( loadData(dir)[3][j] == "red" ):
            #     labels.append(1)
            # elif( loadData(dir)[3][j] == "pink" ):
            #     labels.append(1)
            # elif( loadData(dir)[3][j] == "white" ):
            #     labels.append(1)
            # elif( loadData(dir)[3][j] == "blue" ):
            #     labels.append(1)
            # elif( loadData(dir)[3][j] == "Ibuphil 600 mg" ):
            #     labels.append(1)
            # else:
            #     labels.append(0)
            
        for j in range(len(loadData(dir)[0])):
            image = cv.imread(dir +"/"+ str(file[j]))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # print(xMin[j], xMax[j], yMin[j], yMax[j])
            image = cut_random( image, (224, 224), xMin[j], xMax[j], yMin[j], yMax[j] )
            new_data.append( image )
            labels.append(0)

            # print(labels[j])
            # saveCroppedImage( image, 'nopills', "nopill"+ "_" +str(labels[j]) + "_" + str(loadData(dir)[0][j])) # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

    return np.array(new_data), np.array(labels)

def saveCroppedImage( croppedImage, filePath, imageName ):
    im = Image.fromarray(croppedImage)
    im.save(filePath + '/' + imageName)



pill_name_dict = {
    'Xyzall 5mg': 1,
    'Cipro 500': 2,
    'Ibuphil Cold 400-60': 3,
    'red': 4,
    'pink': 5,
    'white': 6,
    'blue': 7,
    'Ibuphil 600 mg': 8
    
}




# for i in range(len(loadData()[0])):
    

