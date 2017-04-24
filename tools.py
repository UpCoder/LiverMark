# -*- coding: utf-8 -*-
import numpy as np
from skimage import feature, measure
from readFile import readSingleFile as rf
import cv2
def image_binary_slice(image):
    [m, n] = np.shape(image)
    binary_image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if image[i, j] != 0:
                binary_image[i, j] == np.int(255)
            else:
                binary_image[i, j] == np.int(0)
    return binary_image
def find_image_border(image):
    binary_image = image_binary_slice(image)
    border_image = feature.canny(binary_image)
    return border_image
# 传入的是图像
def find_center_points(image):

    binary_image = image_binary_slice(image)
    binary_image.dtype = np.int
    region_props = measure.regionprops(binary_image)
    if len(region_props) == 0:
        return [0, 0]
    centroid = region_props[0].centroid()
    print centroid
    return centroid
def find_intersting_area(image):
    image[np.where(image != 0)] = 1
    region_props = measure.regionprops(image)
    if len(region_props) == 0:
        return [0, 0, 0]
    sum = 0
    for i in range(len(region_props)):
        sum += region_props[0].perimeter
    return [region_props[0].centroid[0], region_props[0].centroid[1]]
# 计算两幅图像的相似度
def measure_similarity(image_x, image_y):
    # 计算质心之间的距离，太慢了
    # centroid_x = find_center_points(image_x)
    # centroid_y = find_center_points(image_y)
    # return (centroid_x[0] - centroid_y[0]) ** 2 + (centroid_x[1] - centroid_y[1]) ** 2

    # 计算面积
    props_x = find_intersting_area(image_x)
    propx_y = find_intersting_area(image_y)
    return calu_props_dist(props_x, propx_y)
def calu_props_dist(props_1, props_2):
    dists = []
    # dists.append((props_1[0] - props_2[0]) ** 2) # 利用周长进行计算
    dists.append((props_1[0] - props_2[0]) ** 2 + (props_1[1] - props_2[1]) ** 2)
    return dists
# 二位数组的归一化
def myNormalization(image,start,end):
    maxValue = np.max(image)
    minValue = np.min(image)
    maxDiff = end - start
    [m, n] = np.shape(image)
    for i in range(m):
        for j in range(n):
            curRate = float((image[i, j]-minValue))/(float(maxValue-minValue))
            # print 'curRate is', curRate
            image[i, j] = maxDiff * curRate + start
    return image
# 一维数组的归一化
def myNormalization1(image,start,end):
    maxValue = np.max(image)
    minValue = np.min(image)
    maxDiff = end - start
    m = len(image)
    for i in range(m):
        curRate = float((image[i]-minValue))/(float(maxValue-minValue))
        # print curRate
        image[i] = maxDiff * curRate + start
    return image
def registration(image1, image2):
    [o, n, m] = np.shape(image1)
    [o1, n1, m1] = np.shape(image2)
    res = []
    used = np.ones((1, o1))  # 标记ART对应的层是否已经用过
    for z in range(o):
        cur_nc_image = image1[z, :, :]
        # silimiaritys = []
        min_silimiarity = 99999999
        min_index = -1
        for z1 in range(o1):
            if used[0, z1] == 0:
                # 已经被使用过了
                continue
            cur_art_image = image2[z1, :, :]
            cur_silimiarity = measure_similarity(cur_nc_image, cur_art_image)
            if cur_silimiarity[0] < min_silimiarity:
                min_silimiarity = cur_silimiarity[0]
                min_index = z1
                # silimiaritys.append(measure_similarity(cur_nc_image, cur_art_image))
        # silimiaritys = np.array(silimiaritys)
        # silimiaritys[:, 0] = myNormalization1(silimiaritys[:, 0], 0, 1)
        # silimiaritys[:, 1] = myNormalization1(silimiaritys[:, 1], 0, 1)
        # min_silimiarity = 10
        # min_index = -1
        # start_index = 0
        # for z1 in range(o1):
        #     if used[0, z1] == 0:
        #         # 已经被使用过了
        #         continue
        #     if (silimiaritys[start_index, 0] + silimiaritys[start_index, 1]) < min_silimiarity:
        #         min_silimiarity = (silimiaritys[start_index, 0] + silimiaritys[start_index, 1])
        #         start_index += 1
        #         min_index = z1
        used[0, min_index] = 0
        print 'nc ', (z + 1), ' --> art ', (min_index + 1)
        res.append([z, min_index])
    return res
art_image_path = 'D:\\MedicalImageAll\\Srr033\\Liver\\Liver_Srr033_ART.mhd'
nc_image_path = 'D:\\MedicalImageAll\\Srr033\\Liver\\Liver_Srr033_NC.mhd'
art_image = rf(art_image_path)
nc_image = rf(nc_image_path)
print np.shape(nc_image)
registration(nc_image, art_image)
