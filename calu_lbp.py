import numpy as np
from skimage.feature import local_binary_pattern
import pywt as wt
from skimage.feature import greycoprops, greycomatrix
def calu_lbp(image, size):
    [y,x] = np.shape(image)
    r = 3
    points = 3*8
    lbp = local_binary_pattern(image, points, r, 'uniform')
    # print 'max Diff is ',np.max(lbp)-np.min(lbp)
    hists = np.histogram(lbp, size)

    allSum = np.sum(hists[0][0:-2])
    energy = []
    for j in range(len(hists[0])-1):
        energy.append(float((hists[0][j] * 1.0)/(allSum*1.0)))

    # print 'caluLBP size is ', np.shape(result)
    return energy
def calu_pixel_value(image, global_max):
    feature = []
    [m, n] = np.shape(image)
    for i in range(m):
        for j in range(n):
            feature.append((1.0*image[i, j])/(1.0*global_max))
    return feature
def caluSingleWavelet(data,n):
    #data = np.ones((4, 4), dtype=np.float64)
    result = []
    for i in range(n):
        # print 'data size is '
        # print np.shape(data)
        coffess = wt.dwt2(data,'haar')
        CA,(CH,CV,CD) = coffess
        # print 'CA size is '
        # print np.shape(CA)
        result.append(caluWaveletPacketEnergy(CA))
        result.append(caluWaveletPacketEnergy(CH))
        result.append(caluWaveletPacketEnergy(CV))
        result.append(caluWaveletPacketEnergy(CD))
        data = CA
    # print 'calu Wavelet size is ',np.shape(result)
    return result
def caluWaveletPacketEnergy(image):
    # print np.shape(image)
    [m, n] = np.shape(image)
    res = 0.0
    for i in range(m):
        for j in range(n):
            res += (image[i,j]**2)
    res = (1.0*res)/(1.0*m*n)
    return res
def calu_glcm_feature(image):
    image = np.array(image)
    min_value = np.min(image)
    if min_value < 0:
        image -= min_value
    glcm_matrixs = greycomatrix(image,
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                [0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2, 5 * np.pi / 8, 3 * np.pi / 4, 7 * np.pi / 8],
                                levels=255)

    contrast_matrixs = greycoprops(glcm_matrixs, 'contrast')
    dissimilarity_matrixs = greycoprops(glcm_matrixs, 'dissimilarity')
    homogeneity_matrixs = greycoprops(glcm_matrixs, 'homogeneity')
    asm_matrixs = greycoprops(glcm_matrixs, 'ASM')
    energy_matrixs = greycoprops(glcm_matrixs, 'energy')
    correlation_matrixs = greycoprops(glcm_matrixs, 'correlation')
    [m, n] = np.shape(contrast_matrixs)
    feature = []
    feature.extend(np.reshape(contrast_matrixs, (1, m*n))[0])
    feature.extend(np.reshape(dissimilarity_matrixs, (1, m*n))[0])
    feature.extend(np.reshape(homogeneity_matrixs, (1, m*n))[0])
    feature.extend(np.reshape(asm_matrixs, (1, m*n))[0])
    feature.extend(np.reshape(energy_matrixs, (1, m*n))[0])
    feature.extend(np.reshape(correlation_matrixs, (1, m*n))[0])
    return feature
def calu_position_feature(image):
    feature = []
    [m, n] = np.shape(image)
    for i in range(m):
        for j in range(n):
            feature.append((i*1.0) / 512.0)
            feature.append((j * 1.0) / 512.0)
    return feature
