# -*- coding: utf-8 -*-
import readFile as rf
import numpy as np
from calu_lbp import calu_lbp as lbp
from calu_lbp import calu_pixel_value as single_pixel_value
from sklearn.neighbors import KNeighborsClassifier as knn_classifier
from calu_lbp import calu_glcm_feature as glcm
import scipy.io as scio
import datetime
def find_z(mask_image):
    [z, n, m] = np.shape(mask_image)
    for i in range(z):
        if np.sum(mask_image[i, :, :]) != 0:
            return i
    return -1
def mask_all_slice_image(image, mask_image, patch_size, feature, label, masked_slice_index, output_mask_name, border_indexs):
    [o, n, m] = np.shape(image)
    knn_k = 1
    knn_model = knn_classifier(n_neighbors=knn_k, n_jobs=8)
    knn_model.fit(feature, label)
    start_time = datetime.datetime.now()
    features = []
    border_x1 = int(0.99 * border_indexs[0])
    border_y1 = int(0.99 * border_indexs[1])
    border_x2 = int(1.01 * border_indexs[2])
    border_y2 = int(1.01 * border_indexs[3])
    # need add condition to keep value from outside
    for z in range(o):
        if z == masked_slice_index:
            continue
        cur_image = image[z, :, :]
        print 'execute z is ', z
        global_max_value = np.max(cur_image)
        cur_slice_feature = []
        for y in range(border_x1, border_x2):
            for x in range(border_y1, border_y2):
                if cur_image[y, x] == 0:
                    continue
                cur_patch = get_patch(cur_image, y, x, patch_size)
                # cur_feature_glcm = glcm(cur_patch)
                cur_feature_lbp = lbp(cur_patch, patch_size)
                cur_feature_pixel = single_pixel_value(cur_patch, global_max_value)
                cur_feature = []
                # ur_feature.extend(cur_feature_glcm)
                cur_feature.extend(cur_feature_pixel)
                cur_feature.extend(cur_feature_lbp)
                cur_feature.append((y * 3.0) / 512.0)
                cur_feature.append((x * 3.0) / 512.0)
                cur_slice_feature.append(cur_feature)
        cur_slice_feature = np.array(cur_slice_feature)
        print 'cur_slice_feature shape is ', np.shape(cur_slice_feature)
        if len(cur_slice_feature) == 0:
            continue
        # glcm的归一化
        # start_index = 0
        # range_num = 80
        # for i in range(6):
        #     cur_slice_feature[:, start_index:start_index + range_num] = myNormalization(
        #         cur_slice_feature[:, start_index:start_index + range_num],
        #         0,
        #         1)
        #     start_index += 80
        features.extend(cur_slice_feature)
    features = np.array(features)
    print 'z = ', z, 'start to predicted, predicted num is ', len(features)
    predicted_res = knn_model.predict(features)
    predicted_index = 0
    for z in range(o):
        cur_image = image[z, :, :]
        if z == masked_slice_index:
            continue
        for y in range(border_x1, border_x2):
            for x in range(border_y1, border_y2):
                if cur_image[y, x] == 0:
                    continue
                mask_image[z, y, x] = predicted_res[predicted_index]
                predicted_index += 1
    end_time = datetime.datetime.now()
    print 'finish z is ', z, ' cost time is ', (end_time - start_time)
    rf.saveSingleFile(mask_image, output_mask_name)
def get_border_indexs(mask_image):
    mask_image = np.array(mask_image)
    find_res = np.where(mask_image == 1)
    x1 = np.min(find_res[0])
    x2 = np.max(find_res[0])
    y1 = np.min(find_res[1])
    y2 = np.max(find_res[1])
    return [x1, y1, x2, y2]
def extract_training(image, mask_image, patch_size):
    features = []
    labels = []
    mask_image = np.array(mask_image)
    image = np.array(image)
    masked_slice_index = find_z(mask_image)
    if masked_slice_index == -1:
        print 'have not marked anyone slice. can not finish the task!'
        return False
    masked_slice_image = image[masked_slice_index, :, :]
    masked_slice_mask_image = mask_image[masked_slice_index, :, :]
    [m, n] = np.shape(masked_slice_image)
    global_max_value = np.max(masked_slice_image)
    for i in range(patch_size/2, m - patch_size/2):
        for j in range(patch_size/2, n - patch_size/2):
            if masked_slice_image[i, j] == 0:
                continue
            if ok_x_y(masked_slice_image, i, j):
                cur_patch = get_patch(masked_slice_image, i, j, patch_size)
                cur_feature_lbp = lbp(cur_patch, patch_size)
                cur_feature_pixel = single_pixel_value(cur_patch, global_max_value)
                # cur_feature_glcm = glcm(cur_patch)
                cur_feature = []
                # cur_feature.extend(cur_feature_glcm)
                cur_feature.extend(cur_feature_pixel)
                cur_feature.extend(cur_feature_lbp)
                cur_feature.append((i * 3.0) / 512.0)
                cur_feature.append((j * 3.0) / 512.0)
                cur_label = masked_slice_mask_image[i, j]
                features.append(cur_feature)
                labels.append(cur_label)
    features = np.array(features)
    # glcm归一化
    # start_index = 0
    # range_num = 80
    # for i in range(6):
    #     features[:, start_index:start_index + range_num] = myNormalization(features[:, start_index:start_index + range_num],
    #                                                                    0,
    #                                                                    1)
    #     start_index += 80
    return np.array(features), np.array(labels), masked_slice_index, get_border_indexs(masked_slice_mask_image)
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
def ok_x_y(image, x, y):
    dirs = [[1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1]]
    if image[x, y] != 0:
        return True
    for i in range(len(dirs)):
        new_x = x + dirs[i][0]
        new_y = y + dirs[i][1]
        if image[new_x][new_y] != 0:
            return True
    return False
def get_patch(image, x, y, patch_size):
    res = []
    for i in range(x-patch_size/2, x+patch_size/2+1):
        single_row = []
        for j in range(y-patch_size/2, y+patch_size/2+1):
            single_row.append(image[i, j])
        res.append(single_row)
    return np.array(res)
image_path = 'D:\\MedicalImageAll\\Srr033\\Liver\\Liver_Srr033_NC.mhd'
mask_path = 'D:\\MedicalImageAll\\Srr033\\TumorMask_New\\TumorMask_Srr033_NC_1.mhd'
# image_path = 'E:\\work\\LiverMask\\data\\Liver_Srr000_ART.mhd'
# mask_path = 'E:\\work\\LiverMask\\data\\TumorMask_Srr000_ART.mhd'
image = rf.readSingleFile(image_path)
mask_image = rf.readSingleFile(mask_path)
patch_size = 31
features, labels, masked_slice_index, border_indexs = extract_training(image, mask_image, patch_size)
scio.savemat('output.mat', {'features': features,
                            'labels': labels,
                            'masked_slice_index': masked_slice_index,
                            'border_indexs': border_indexs})
print 'features min value is ', np.min(features), 'features max value is ', np.max(features)
print 'features size is ', np.shape(features)
print 'labels size is ', np.shape(labels)
print 'all labels sum is ', np.sum(labels)
print 'masked_slice_index is ', masked_slice_index
mask_all_slice_image(image,
                     mask_image,
                     patch_size,
                     feature=features,
                     label=labels,
                     masked_slice_index=masked_slice_index,
                     output_mask_name='E:\\work\\LiverMask\\data\\TumorMask_Srr033_NC_1.mhd',
                     border_indexs=border_indexs)
# E:\\work\\LiverMask\\data\\TumorMask_Srr033_ART_1.mhd 是不使用坐标特征，并且patch_size = 37的结果。
# E:\\work\\LiverMask\\data\\TumorMask_Srr033_ART_1.mhd 是使用坐标特征，并且patch_size = 31的结果。