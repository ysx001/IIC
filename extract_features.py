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
    img = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))

    tmp = extract_feature(model, img)
    print(tmp.shape)
    print(tmp)