#!python
#!/usr/bin/env python
from scipy.io import loadmat, savemat
from glob import glob
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str,
                    default='/bmrNAS/people/yuxinh/DL_diffseg/DiffSeg-Data')

config = parser.parse_args()
root = config.root

import csv
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

def _hungarian_match(flat_preds, flat_targets, num_k):
    num_samples = flat_targets.shape[0]
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # print(num_correct)
    # print(num_samples - num_correct)
    # num_correct is small
    match = linear_assignment(num_samples - num_correct)
    # return as list of tuples, out_c to gt_c
    match_dict = {}
    for out_c, gt_c in match:
        match_dict[out_c] = gt_c
    print(match_dict)
    return match_dict

def _dice(im1, im2, numk):
    result = [0] * numk
    for k in range(numk):
        print(k)
        print("Numerator: ", np.sum(im1[im2==k]==k) * 2.0)
        print(("Denomator: ", np.sum(im1[im1==k]==k) + np.sum(im2[im2==k]==k)))
        result[k] = np.sum(im1[im2==k]==k) * 2.0 / (np.sum(im1[im1==k]==k) + np.sum(im2[im2==k]==k))
    return result

def reorder(pred, match_dict):
    reordered_pred = copy(pred)
    for i in range(len(reordered_pred)):
        for j in range(len(reordered_pred[0])):
            if reordered_pred[i, j] in match_dict:
                reordered_pred[i, j] = match_dict[reordered_pred[i, j]]
            else:
                reordered_pred[i, j] = 5
    return reordered_pred

def getHeatMap():
    # %% Read the names of the labels and write them together with counts
    import csv
    from copy import copy
    import numpy as np
    subjects = sorted(glob(osp.join(root, 'mwu*')))
    heatMap = np.zeros((90, 108, 5))
    sliceCount = 0 
    for subject_id in subjects:
        print(subject_id)
        image_mat = loadmat(osp.join(root, subject_id, "combined_data.mat"))
        sliceCount += image_mat['combinedsegs'].shape[2]
        labels = image_mat['combinedsegs'][:, :, :, 1]
        for labelId in range(5):
            filtered = np.where(labels == labelId, 1, 0)
            filtered = np.sum(filtered, axis=2)
            heatMap[:, :, labelId] += filtered
    # average
    heatMap /= (sliceCount * 1.0)
    savemat(osp.join(root, "heapMap.mat"), {
        'imgs': image_mat['imgs'],
        'heatMap': heatMap
    })
    return heatMap

subjects = sorted(glob(osp.join(root, 'mwu*')))
k_means_classes = 5

dice_mat = np.zeros((len(subjects), 90, 6, 5))
acc_mat = np.zeros((len(subjects), 90, 6))
heatMap = np.zeros((80, 98, 90, 6, 5))
sliceCount = 0 

subj_idx = 0
for subject_id in subjects:
    print(subject_id)
    image_mat = loadmat(osp.join(root, subject_id, "kmeans.mat"))
    sliceCount += image_mat['m0'].shape[2]
    for s in range(image_mat['m0'].shape[2]):
        target = image_mat['m0'][:, :, s]
        target_crop = target[4: 84, 5:103]
        preds = []
        preds.append(image_mat['m1'][:, :, s] - 1)         
        preds.append(image_mat['m2'][:, :, s] - 1)
        preds.append(image_mat['m3'][:, :, s] - 1)
        preds.append(image_mat['m4'][:, :, s] - 1)
        preds.append(image_mat['m5'][:, :, s] - 1)
        preds.append(image_mat['m6'][:, :, s] - 1)
        p_idx = 0
        for pred in preds:
            pred_crop = pred[4:84, 5:103]
            reordered_pred_crop = reorder(pred_crop, _hungarian_match(pred_crop.flatten(), target_crop.flatten(), 5))
            # reordered_pred = reorder(pred, _hungarian_match(pred.flatten(), target.flatten(), 5))
            print(reordered_pred_crop.shape, target_crop.shape)
            # print(reordered_pred.shape, target.shape)
            acc_crop = int((reordered_pred_crop.flatten() == target_crop.flatten()).sum()) / float(target_crop.flatten().shape[0])
            # acc = int((reordered_pred.flatten() == target.flatten()).sum()) / float(target.flatten().shape[0])
            # print(acc, acc_crop)
            dice_crop = _dice(reordered_pred_crop, target_crop, k_means_classes)
            # dice = _dice(reordered_pred, target, k_means_classes)
            dice_mat[subj_idx, s, p_idx] = dice_crop
            acc_mat[subj_idx, s, p_idx] = acc_crop
            for labelId in range(5):
                heatMap[:, :, s, p_idx, labelId] += np.where(reordered_pred_crop == labelId, 1, 0)
            p_idx += 1
    subj_idx += 1
    break

heatMap = np.sum(heatMap, axis=2)
heatMap /= (sliceCount * 1.0)
    
savemat(osp.join(root, "scores.mat"), {
        'dice': dice_mat,
        'acc': acc_mat,
        "heatMap": heatMap
})

# k=1

# # segmentation
# seg = np.zeros((100,100), dtype='int')
# seg[30:70, 30:70] = k

# # ground truth
# gt = np.zeros((100,100), dtype='int')
# gt[30:70, 40:80] = k

# dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))

# print 'Dice similarity score is {}'.format(dice)


############ dice socre #############

