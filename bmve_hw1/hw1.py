#! /usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage
import os
import sift, ransac
from PIL import Image
# import cv
# import cv2

def rgb2gray(I):
    return np.sum(I*np.array([0.299, 0.587, 0.114]), axis=2)


def compute_sift(I):
    """
    compute sift descriptors of an image
    
    Parameters:
    -----------
    I : image
        MxN (grayscale) or MxNx3 (rgb) image
        
    Returns:
    --------
    sift_data : tuple of (locs, descriptors)
        locs is a Nx4 matrix that contains in each row the y,x position and the scale and orientation of a feature
        
        descriptors is a Nx128 matrix that contains in each row a 128.dimensional feature descriptor
    """
    if len(I.shape) not in (2,3):
        raise RuntimeError('expected one or three channel image')
    if len(I.shape) == 3 and I.shape[2] != 3:
        raise RuntimeError('only three channel images are supported')

    if len(I.shape) == 3:
        Ig = rgb2gray(I)
    else:
        Ig = I

    # the sift binary expects a *.pgm file
    scipy.misc.imsave('temp.pgm', np.round(Ig).astype('uint8'))
    sift.process_image('temp.pgm', 'sift.key')
    sift_data = sift.read_features_from_file('sift.key')
    os.remove('temp.pgm')
    os.remove('sift.key')

    return sift_data

def backward_mapping_fusion(I, I2, H):
    corners = np.inner(H, np.array([[0, 0, 1], [0, 340, 1], [512, 0, 1], [512, 340, 1]])) # Positions hard coded!
    corners /= corners[2]
    min_w = int(round(min(corners[0])))
    max_w = int(round(max(corners[0])))
    min_h = int(round(min(corners[1])))
    max_h = int(round(max(corners[1])))

    width, height = max_w-min_w, max_h-min_h
    Inew = np.zeros((height, width, 3))
    H_I = np.linalg.inv(H)

    cords = np.zeros((3, height*width))
    for i in xrange(min_h, max_h):
        for j in xrange(min_w, max_w):
            cord = (H_I.dot([float(j), float(i), 1.]))
            cords[:, (i-min_h)*width+(j-min_w)] = cord / cord[2]

    for i in xrange(3):
        Inew[:, :, i] = scipy.ndimage.map_coordinates(I[:, :, i], [[cords[1, :]], [cords[0, :]]]).reshape(Inew.shape[:2])

    # Fusion of the pictures!
    max_min_w = max(min_w, 0)
    max_min_h = max(min_h, 0)
    I2_rows, I2_cols = I2.shape[:2]
    fir_edge = int(abs(min(min_h, 0)))
    sec_edge = int(max(max_h - I2_rows, 0))
    thi_edge = int(abs(min(min_w, 0)))
    fou_edge = int(max(max_w - I2_cols, 0))
    Ifusion = np.lib.pad(I2, [[fir_edge, sec_edge], [thi_edge, fou_edge], [0, 0]], mode='constant')

    Inew_rows, Inew_cols = Inew.shape[:2]
    for i in xrange(Inew_rows):
        for j in xrange(Inew_cols):
            if Inew[i, j, 0] != 0 and Inew[i, j, 1] != 0 and Inew[i, j, 2] != 0:
                Ifusion[i+max_min_h, j+max_min_w] = Inew[i, j]

    return Ifusion

def calc_hXY(I1, I2):
    sift1 = compute_sift(I1)
    sift2 = compute_sift(I2)
    match = sift.match(sift1[1], sift2[1])

    x1 = []; x2 = [] # build lists of corresponding points
    for idx,m in enumerate(match):
        if m>0:
            x1.append(sift1[0][idx,0:2])
            x2.append(sift2[0][int(m),0:2])

    x1 = np.fliplr(np.array(x1)) # sift returns (y,x) coordinates
    x2 = np.fliplr(np.array(x2))

    model = ransac.Homography_Model() # fit homography
    n = 5                       # minimum number of points required to fit a model
    iters = 1000                # ransac iterations
    error_threshold = 0.3       # error threshold (distance in pixels) above which a point is considered an outlier
    num_inliers = x1.shape[0]/3 # required number of points that have error < error_threshold so the model is considered a good fit
    # we want that our model has at least one third of the data as inliers

    return ransac.ransac(np.hstack((x1,x2)), model, n, iters, error_threshold, num_inliers)

if __name__ == '__main__':
    plt.close('all')

    I1 = scipy.misc.imread('cathedral1.png')
    I2 = scipy.misc.imread('cathedral2.png')
    I3 = scipy.misc.imread('cathedral3.png')

    H21 = calc_hXY(I2, I1)
    I12 = backward_mapping_fusion(I2, I1, H21)

    H312 = calc_hXY(I3, I12)
    I123 = backward_mapping_fusion(I3, I12, H312)

    plt.figure('I12')
    plt.imshow(I12.astype('uint8'), interpolation='none')
    plt.figure('I123')
    plt.imshow(I123.astype('uint8'), interpolation='none')
    plt.show()
