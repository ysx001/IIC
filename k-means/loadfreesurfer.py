#!python
#!/usr/bin/env python
from scipy.io import loadmat, savemat
from glob import glob
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str,
                    default='/home/sarah/cs221/project/DiffSeg-Data')

config = parser.parse_args()
root = config.root
# print(len(subjects))
def count_labels():
# %% Write the labels acutally in data to labels.csv
    subjects = sorted(glob(osp.join(root, 'mwu*')))

    actual_labels = {} # key: labels, value: counts
    for subject_id in subjects:
        image_mat = loadmat(osp.join(root, subject_id, 'jpg'))
        for s in range(image_mat['segs'].shape[2]):
            label = image_mat['segs'][:, :, s, 1]
            for i in range(len(label)):
                for j in range(len(label[0])):
                    if label[i, j] not in actual_labels:
                        actual_labels[label[i, j]] = 1
                    else:
                        actual_labels[label[i, j]] += 1

    import csv
    w = csv.writer(open(osp.join(root, "labels.csv"), "w"))
    for key, val in actual_labels.items():
        w.writerow([key, val])

    print(len(actual_labels))
    print(actual_labels)

def labelNameCount():
    # %% Read the names of the labels and write them together with counts
    label_names = {}
    count = 0
    with open("/home/sarah/IIC/code/datasets/segmentation/FreeSurferColorLUT.txt") as f:
        for line in f:
            vals = line.split()
            if len(vals) > 2 and vals[0].isdigit():
                count+=1
                label_names[vals[0]] = vals[1]

    print (count)
    import csv

    w = csv.writer(open(osp.join(root, "labelNameCount.csv"), "w"))
    index = 0
    with open(osp.join(root, "labels.csv")) as label_counts:
        reader = csv.reader(label_counts)
        for rows in reader:
            label = rows[0]
            count = rows[1]
            name = label_names[label]
            w.writerow([label, index, count, name])
            index += 1

def combineLabelFiveCategory():
    # %% Read the names of the labels and write them together with counts while combining based on category
    label_names = {}
    count = 0
    with open("FreeSurferColorLUT.txt") as f:
        for line in f:
            vals = line.split()
            if len(vals) > 2 and vals[0].isdigit():
                count+=1
                label_names[vals[0]] = vals[1]

    import csv
    from collections import defaultdict

    w = csv.writer(open(osp.join(root, "combinedLabels.csv"), "w"))
    index = 0
    wm = [2, 41, 77, 7, 46, 251, 252, 253, 254, 255] # from mri_binarize https://surfer.nmr.mgh.harvard.edu/fswiki/mri_binarize#Example1
    # below combinations from tqy
    wmNames = ['vessel', 'Optic-Chiasm']
    gmNames = ['non-WM-hypointensities', 'ctx', 'Cortex']
    dgmNames = ['Pallidum', 'Hippocampus', 'Accumbens', 'Caudate', 'Thalamus', \
        'Amygdala', 'Putamen', 'Brain-Stem']
    csfNames = ['choroid-plexus', 'Vent', 'CSF']
    labelCount_dict = defaultdict(int)
    label_dict = defaultdict(list)
    labelToIdx = {}


    with open(osp.join(root, "labels.csv")) as label_counts:
        reader = csv.reader(label_counts)
        for rows in reader:
            label = rows[0]
            count = rows[1]
            if int(label) in wm or sum([name in label_names[label] for name in wmNames]) == 1:
                labelCount_dict['WhiteMatter'] += int(count)
                label_dict['WhiteMatter'].append(label)
            elif sum([name in label_names[label] for name in csfNames]) == 1:
                labelCount_dict['CSF'] += int(count)
                label_dict['CSF'].append(label)
            elif sum([name in label_names[label] for name in dgmNames]) == 1:
                labelCount_dict['DeepGreyMatter'] += int(count)
                label_dict['DeepGreyMatter'].append(label)
            elif sum([name in label_names[label] for name in gmNames]) == 1:
                labelCount_dict['GreyMatter'] += int(count)
                label_dict['GreyMatter'].append(label)
            else:
                name = label_names[label]
                w.writerow([index, name, count, label])
                labelToIdx[frozenset([label])] = index
                index += 1
        for name in label_dict:
            w.writerow([index, name, labelCount_dict[name], label_dict[name]])
            labelToIdx[frozenset(label_dict[name])] = index
            index += 1
    return labelToIdx


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

import matplotlib.pyplot as plt
# heatMap = getHeatMap()
# heatMap = loadmat(osp.join(root, "heapMap.mat"))
# heatMap = loadmat(osp.join(root, "scores.mat"))
heatMap = loadmat("scores.mat")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
m = 4

ax1.imshow(heatMap['heatMap'][:, :, m, 1])
ax1.set_title('DeepGreyMatter')
ax2.imshow(heatMap['heatMap'][:, :, m, 2])
ax2.set_title('WhiteMatter')

ax3.imshow(heatMap['heatMap'][:, :,m, 3])
ax3.set_title('GreyMatter')
ax4.imshow(heatMap['heatMap'][:, :, m, 4])
ax4.set_title('CSF')

plt.show()


# # %% Read the names of the labels and write them together with counts
# import csv
# import matplotlib.pyplot as plt
# from copy import copy
# import numpy as np
# subjects = sorted(glob(osp.join(root, 'mwu*')))
# # labelToIdx = combineLabelFiveCategory()

# for subject_id in subjects:
#     print(subject_id)
#     image_mat = loadmat(osp.join(root, subject_id, "data.mat"))
#     newSliceLabels = copy(image_mat['segs'][:, :, :, 1])
#     print(newSliceLabels.shape)
    # for s in range(image_mat['segs'].shape[2]):
    #     labelCopy = copy(image_mat['segs'][:, :, s, 1])
    #     for i in range(len(labelCopy)):
    #         for j in range(len(labelCopy[0])):
    #             for labels in labelToIdx:
    #                 if str(labelCopy[i, j]) in labels:
    #                     # print(labelCopy[i, j], labels)
    #                     labelCopy[i, j] = labelToIdx[labels]
    #     newSliceLabels[:, :, s] = labelCopy
    
    # newSeg = np.append(image_mat['segs'][:, :, :, 1, None], newSliceLabels[:, :, :, None], axis=3)    
    # savemat(osp.join(root, subject_id, "combined_data.mat"), {
    #     'imgs': image_mat['imgs'],
    #     'combinedsegs': newSeg
    # })
    # # print(newSliceLabels.shape)
    # s = 50 
    # print(newSeg.shape)
    # print(newSeg[:, :, s, 0].min(), newSeg[:, :, s, 0].max())
    # print(newSeg[:, :, s, 1].min(), newSeg[:, :, s, 1].max())
    # fig, (ax1, ax2) = plt.subplots(nrows=2)
    # ax1.imshow(newSeg[:, :, s, 0], vmin=0, vmax=2033)
    # ax1.set_title('Original Freesurfer Segmentation')
    # ax2.imshow(newSeg[:, :, s, 1])
    # ax2.set_title('Combined into 5 classes')
    # plt.show()
    # break



# # plt.show()

# print(x['imgs'][:, :, slide, 1].min(), x['imgs'][:, :, slide, 1].max())
# axarr[0,0].imshow(x['imgs'][:, :, slide, 0])
# axarr[0,1].imshow(x['imgs'][:, :, slide, 1])
# axarr[1,0].imshow(x['imgs'][:, :, slide, 2])
# axarr[1,1].imshow(x['imgs'][:, :, slide, 3])
# # axarr[2,0].imshow(x['segs'][:, :, slide, 0],  cmap='plasma', vmin=0, vmax=77)
# axarr[2,0].imshow(x['segs'][:, :, slide, 1],  cmap='plasma', vmin=0, vmax=2033)
# axarr[2,1].imshow(label,  cmap='plasma')

# # plt.colorbar()
# plt.show()

# %%