from __future__ import print_function

import os.path as osp
import pickle
from glob import glob
import csv

from PIL import Image
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as tvt
from torch.utils import data
from skimage.transform import resize

NUM_SLICES = 90

class DiffSeg(data.Dataset):
  """Base class
  This contains fields and methods common to all Mri datasets:
  DiffSeg
  """
  def __init__(self, root, partition = 0, transform_img=None, transform_label=None):
    super(DiffSeg, self).__init__()
    self.root = root
    self.transform_img = transform_img
    self.transform_label = transform_label

    subjects = sorted(glob(osp.join(self.root, 'mwu*')))
    if partition == 0:
        subjects = subjects[0: len(subjects) // 2]
    else:
        subjects = subjects[len(subjects) // 2 : len(subjects)]
    self.files = subjects

  def __getitem__(self, index):
    subject_idx = index // NUM_SLICES
    slice_idx = index % NUM_SLICES
    # print(subject_idx, slice_idx, index)
    subject_id = self.files[subject_idx]
    image, label = self._load_data(subject_id, slice_idx)
    if self.transform_img:
      image = self.transform_img(image)
    if self.transform_label:
      label = self.transform_label(label)

    return image, label

  def __len__(self):
    return len(self.files) * NUM_SLICES

  def _load_data(self, subject_id, slice_idx):
    image_mat = sio.loadmat(osp.join(self.root, subject_id, "combined_data.mat"))
    slices = sorted(glob(osp.join(self.root, subject_id, 'jpg', "im*")))

    # shape (90, 108, 90, 4)
    # each slice is 90 * 108
    # 90 slices per subject
    # 4 channels, each channel representing b=0, dwi, md and fa
    image = Image.open(slices[slice_idx])
    # using the combined segmentations with 5 classes
    label = image_mat["combinedsegs"][:, :, slice_idx, 1].astype(dtype = 'uint8')
    # print(label.shape)
    label = resize(label, (224, 224))
    label = np.expand_dims(label, axis=0)
    label = np.concatenate([np.where(label == 0, 1, 0), np.where(label == 1, 2, 0), np.where(label == 2, 3, 0), np.where(label == 3, 4, 0), np.where(label == 4, 5, 0)], axis=0)
    # print(label.shape)
    return image, label

