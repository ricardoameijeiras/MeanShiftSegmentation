import os
import pathlib
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from numpy import asarray
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage import color
from plotclusters3D import plotclusters3D, plotclusters3D_rgb


# findpeak optimized, necessary for the second speed-up
def findpeak_opt(data, idx, r, c, features_size):
    t = 0.01
    shift = np.array([1])
    dataT = data.T
    data_point = data[:, idx]
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, features_size)
    cpts = np.zeros(len(data.T))

    # Runs until the shift is smaller than the set threshold
    while shift.all() > t:
        # Calculate the euclidian distance between the current data point and
        # the rest of data (pairwise)
        dist = cdist(dataT, data_pointT)

        # We obtain where there are datapoints within our given radius - window.
        inds = np.where(dist <= r)
        datapoints_in_range = dataT[inds, :]

        # Points within a distance of r/c of the search path are obtained, and
        # a value of 1 is given
        cpts_idx = np.where(dist <= r/c)[0]
        cpts[cpts_idx] = 1

        # Peak is defined as the mean of every datapoint in range
        peak = np.mean(datapoints_in_range, axis=1)

        # We define our shift and data point is updated to get convergence
        shift = np.abs(data_pointT - peak[0])
        data_pointT = peak[0].reshape(1, features_size)

    return data_pointT.T, cpts


# Mean shift algorithm with two speed-ups
def meanshift_opt_2(data, r, c, features):
    labels = np.zeros(len(data.T))
    peaks = []
    label_no = 1

    # findpeak_opt is called for the first index out of the loop, to initialize peaks
    peak, cpts = findpeak_opt(data, 0, r, c, features)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # Every data point is iterated through but will be ignored in case it's already
    # found in the labels list
    for idx in range(0, len(data.T)):
        if idx % 5000 == 0:
            print("Index", idx, "/", len(data.T))

        if labels[idx] != 0:
            continue

        # Points within distance r/c of the search path are associated with the
        # converged peak
        new_peak = True
        peak, cpts = findpeak_opt(data, idx, r, c, features)
        associated_points = np.where(cpts == 1)
        peakT = np.concatenate(peak, axis=0).T

        # It is check whether a new peak is found, or in the contrary we can bound
        # the one we found with an existent one
        for i in range(0, label_no):
            distance = cdist(np.array(peaks)[i].reshape(1, features), peakT.reshape(1, features))
            if distance < (r / 2):
                labels[associated_points] = i + 1
                new_peak = False
                break

        # In case it is a peak never seen before, we augment the number of labels,
        # append a new peak to the list and link its label to the associated points
        if new_peak:
            peaks.append(peakT)
            label_no += 1
            labels[associated_points] = label_no

    return labels.astype(int), np.array(peaks).T[0:3]


# Mean shift algorithm with one speed-up
def meanshift_opt(data, r):
    labels = np.zeros(len(data.T))
    peaks = []
    label_no = 1

    # findpeak is called for the first index out of the loop
    peak, inds = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # First element of the labels list is always 1
    labels[inds[0]] = label_no

    # Every data point is iterated through but will be ignored in case it's already
    # found in the labels list
    for idx in range(0, len(data.T)):
        if labels[idx] != 0:
            continue

        # findpeak gets called for every data point (unless found previously in the
        # labels list)
        new_peak = True
        peak, inds = findpeak(data, idx, r)
        peakT = np.concatenate(peak, axis=0).T

        # In case it is a peak never seen before, we augment the number of labels, append
        # a new peak to the list and link its label to the associated points
        for i in range(0, label_no):
            distance = cdist(np.array(peaks)[i].reshape(1, 3), peakT.reshape(1, 3))
            if distance < (r / 2):
                labels[inds] = i + 1
                new_peak = False
                break

        # In case it is a peak never seen before, we augment the number of labels, append
        # a new peak to the list and link its label to the indices returned from findpeak
        if new_peak:
            peaks.append(peakT)
            label_no += 1
            labels[inds] = label_no

    return labels, np.array(peaks).T


# findpeak function suitable both for 0 speed-ups and one speed-up
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # Runs until the shift is smaller than the set threshold
    while shift.all() > t:
        # Calculate the euclidian distance between the current data point and
        # the rest of data (pairwise)
        dist = cdist(dataT, data_pointT)

        # We obtain where there are datapoints within our given radius - window.
        inds = np.where(dist <= r)
        datapoints_in_range = dataT[inds, :]

        # Peak is defined as the mean of every datapoint in range
        peak = np.mean(datapoints_in_range, axis=1)

        # We define our shift and data point is updated to get convergence
        shift = np.abs(data_pointT - peak[0])
        data_pointT = peak[0].reshape(1, 3)

    return data_pointT.T, inds[0]


# Mean shift algorithm without speed-ups
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = []
    label_no = 1
    labels[0] = label_no

    # findpeak is called for the first index out of the loop
    peak, inds = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # Every data point is iterated through
    for idx in range(0, len(data.T)):
        new_peak = True
        peak, inds = findpeak(data, idx, r)
        peakT = np.concatenate(peak, axis=0).T

        # It is checked whether it is actually a new found peak or can be connected
        # to an already existent one
        for i in range(0, label_no):
            distance = cdist(np.array(peaks)[i].reshape(1, 3), peakT.reshape(1, 3))
            if distance < (r / 2):
                labels[idx] = i + 1
                new_peak = False
                break

        # In case it is a new peak, it gets appended to the list and the number of
        # labels is incremented
        if new_peak:
            peaks.append(peakT)
            label_no += 1
            labels[idx] = label_no

    return labels, np.array(peaks).T


# Returns the image with the coordinate concatenation for 5D features
def conc_coordinates(imglab, img):
    # Pair of coordinates for x and y are created
    x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # Now, these coordinates get reshaped and concatenated to the image matrix
    X = np.reshape(x, (img.shape[0] * img.shape[1], 1))
    Y = np.reshape(y, (img.shape[0] * img.shape[1], 1))
    img_reshaped = np.concatenate((imglab, X, Y), axis=1)

    return img_reshaped


# Method for image segmentation
def segmIm(img, r, c, features):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)

    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))

    if features == 3:
        # Mean shift algorithm function with two speed-ups is called
        labels, peaks = meanshift_opt_2(imglab.T, r, c, features)

        # Labels are reshaped to only one column for easier handling
        labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

        # We iterate through every possible peak and its corresponding label
        for label in range(0, peaks.shape[1]):
            # Obtain indices for the current label in labels array
            inds = np.where(labels_reshaped == label + 1)[0]

            # The segmented image gets indexed peaks for the corresponding label
            corresponding_peak = peaks[:, label]
            segmented_image[inds, :] = corresponding_peak

    elif features == 5:
        # Concatenates the coordinates to imglab
        conc_img = conc_coordinates(imglab, img)

        # Mean shift algorithm function with two speed-ups is called
        labels, peaks = meanshift_opt_2(conc_img.T, r, c, features)

        # Labels are reshaped to only one column for easier handling
        labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

        # We iterate through every possible peak and its corresponding label
        for label in range(0, peaks.shape[1]):
            # Obtain indices for the current label in labels array
            inds = np.where(labels_reshaped == label + 1)[0]

            # The segmented image gets peaks indexed for the corresponding label
            corresponding_peak = peaks[:, label]
            segmented_image[inds, :] = corresponding_peak

    else:
        print("Specified number of features is incorrect")

    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    print("Total number of found peaks:", peaks.shape[1] + 1)

    plotclusters3D_rgb(img_reshaped.T, labels, np.array(peaks.T))

    return color.lab2rgb(segmented_image)


def main(argv):
    # The given parameters get processed
    image = argv[0]
    radius = argv[1]
    feature_type = argv[2]
    c = 4

    path2data = os.path.join(pathlib.Path(__file__).parent.absolute(), 'data/pts.mat')
    path2img = os.path.join(pathlib.Path(__file__).parent.absolute(), image)

    # Mat file and image get loaded
    mat = scipy.io.loadmat(path2data)
    data = np.asarray(mat['data'])
    img = cv2.imread(path2img)

    # Uncomment any of the next tuples in case of desired testing
    #labels, peaks = meanshift_opt_2(data, float(radius), float(c), 3)
    #plotclusters3D(data, labels, peaks)

    #labels, peaks = meanshift_opt(data, float(radius))
    #plotclusters3D(data, labels, peaks)

    #labels, peaks = meanshift(data, float(radius))
    #plotclusters3D(data, labels, peaks)

    # PRE-PROCESSING

    # Normalization
    pixels = asarray(img).astype('float32')
    norm = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))

    # Mean filter
    kernel = np.ones((5, 5), np.float32) / 25
    mean_filtered_img = cv2.filter2D(norm, -1, kernel)

    # Gaussian filter
    gaussian_filtered_img = gaussian_filter(norm, sigma=1)

    # Median blur
    median_filtered_img = cv2.medianBlur(norm, 5)

    # segIm is called, obtaining the segmented image
    segmented_image = segmIm(median_filtered_img, float(radius), float(c), int(feature_type))

    # Mean filtered normalized image is displayed
    cv2.imshow('Input image to segmIm', median_filtered_img)
    cv2.waitKey(0)
    plt.show()

    #Segmented image is finally shown
    cv2.imshow('Segmented image', segmented_image)
    cv2.waitKey(0)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])