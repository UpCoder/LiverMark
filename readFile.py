# -*- coding: utf-8 -*-
from SimpleITK import SimpleITK as itk
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io,exposure
import matplotlib.pyplot as plt
import cv2
from skimage import feature
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
def readSingleFile(filePath):
    header = itk.ReadImage(filePath)
    image = itk.GetArrayFromImage(header)
    print type(image)
    return image
def saveSingleFile(image,fileName):
    print 'image type is ', type(image)
    header = itk.GetImageFromArray(image)
    itk.WriteImage(header, fileName)
def findUsefulImage(image):
    [z, x, y] = np.shape(image)
    res = []
    for i in range(z):
        if sum(sum(image[i, :, :] != 0)) !=0:
            res.append(image[i, :, :])
    print len(res)
    res = np.array(res)
    print type(res)
    return res
def caluROI(image3D):
    indexs = np.where(image3D != 0)
    minX = np.min(indexs[:][2])
    maxX = np.max(indexs[:][2])
    minY = np.min(indexs[:][1])
    maxY = np.max(indexs[:][1])
    minZ = np.min(indexs[:][0])
    maxZ = np.max(indexs[:][0])
    print minX, maxX
    print minZ, maxZ
    imageROI = image3D[minZ:maxZ+1, minY:maxY+1, minX:maxX+1]
    return imageROI

def caluROI2D(image2D):
    indexs = np.where(image2D != 0)
    minY = np.min(indexs[:][1])
    maxY = np.max(indexs[:][1])
    minZ = np.min(indexs[:][0])
    maxZ = np.max(indexs[:][0])
    imageROI = image2D[minZ:maxZ + 1, minY:maxY + 1]
    return imageROI