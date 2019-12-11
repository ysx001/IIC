import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import argparse

from scipy.io import loadmat, savemat
from glob import glob
import os.path as osp
from copy import copy

def meanshift(image):
    X = np.reshape(image, [-1, 1])
    # print(X.shape)
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
    if bandwidth <= 0.0:
        bandwidth += 1
    # print(bandwidth)

    clustering = MeanShift(bandwidth=bandwidth).fit(X)
    labels = clustering.labels_

    cluster_centers = clustering.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    segmented_image = np.reshape(labels, image.shape)
    return segmented_image

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str,
                    default='/bmrNAS/people/yuxinh/DL_diffseg/DiffSeg-Data')
config = parser.parse_args()
root = config.root

subjects = sorted(glob(osp.join(root, 'mwu*')))
for subject_id in subjects:
    print(subject_id)
    image_mat = loadmat(osp.join(root, subject_id, "data.mat"))
    newSliceLabels = copy(image_mat['segs'][:, :, :, 1])
    for s in range(image_mat['imgs'].shape[2]):
        print(s)
        image = image_mat['imgs'][:, :, s, 0]
        segImg = meanshift(image)
        newSliceLabels[:, :, s] = segImg
    savemat(osp.join(root, subject_id, "meanshift.mat"), {
        'meanshift': newSliceLabels
    })

