import os
import numpy as np
import scipy.io as sio
from glob import glob

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
 
from PIL import Image

TARGET_IMG_SIZE = 128
img_to_tensor = transforms.ToTensor()
 
def make_model():
    model=models.vgg16(pretrained=True).features[:28]
    model=model.eval()
    model.cuda()
    return model


def extract_feature(model,img):
    model.eval()

    tensor=img_to_tensor(img)
    tensor=tensor.cuda()
    
    result=model(Variable(tensor))
    result_npy=result.data.cpu().numpy()
    
    return result_npy[0]

def pad_if_too_small(data, sz):
  reshape = (len(data.shape) == 2)
  if reshape:
    h, w = data.shape
    data = data.reshape((h, w, 1))

  h, w, c = data.shape

  if not (h >= sz and w >= sz):
    # img is smaller than sz
    # we are missing by at least 1 pixel in at least 1 edge
    new_h, new_w = max(h, sz), max(w, sz)
    new_data = np.zeros([new_h, new_w, c], dtype=data.dtype)

    # will get correct centre, 5 -> 2
    centre_h, centre_w = int(new_h / 2.), int(new_w / 2.)
    h_start, w_start = centre_h - int(h / 2.), centre_w - int(w / 2.)

    new_data[h_start:(h_start + h), w_start:(w_start + w), :] = data
  else:
    new_data = data
    new_h, new_w = h, w

  if reshape:
    new_data = new_data.reshape((new_h, new_w))

  return new_data

def pad_and_or_crop(orig_data, sz, mode=None, coords=None):
  data = pad_if_too_small(orig_data, sz)

  reshape = (len(data.shape) == 2)
  if reshape:
    h, w = data.shape
    data = data.reshape((h, w, 1))

  h, w, c = data.shape
  if mode == "centre":
    h_c = int(h / 2.)
    w_c = int(w / 2.)
  elif mode == "fixed":
    assert (coords is not None)
    h_c, w_c = coords
  elif mode == "random":
    h_c_min = int(sz / 2.)
    w_c_min = int(sz / 2.)

    if sz % 2 == 1:
      h_c_max = h - 1 - int(sz / 2.)
      w_c_max = w - 1 - int(sz / 2.)
    else:
      h_c_max = h - int(sz / 2.)
      w_c_max = w - int(sz / 2.)

    h_c = np.random.randint(low=h_c_min, high=(h_c_max + 1))
    w_c = np.random.randint(low=w_c_min, high=(w_c_max + 1))

  h_start = h_c - int(sz / 2.)
  w_start = w_c - int(sz / 2.)
  data = data[h_start:(h_start + sz), w_start:(w_start + sz), :]

  if reshape:
    data = data.reshape((sz, sz))

  return data, (h_c, w_c)
    
if __name__=="__main__":
    model=make_model()
    imgpath='/bmrNAS/people/yuxinh/DL_diffseg/DiffSeg-Data'
    subjects = sorted(glob(os.path.join(imgpath, 'mwu*')))
    subject_id = subjects[0]
    slice_idx = 60
    image_mat = sio.loadmat(os.path.join(imgpath, subject_id, "data.mat"))
    # shape (90, 108, 90, 4)
    # each slice is 90 * 108
    # 90 slices per subject
    # 4 channels, each channel representing b=0, dwi, md and fa
    image = image_mat["imgs"][:,:,slice_idx,:]
    img, _  = pad_and_or_crop(image, TARGET_IMG_SIZE, mode="centre")
    print(img.shape)

    tmp = extract_feature(model, img)
    print(tmp.shape)
    print(tmp)