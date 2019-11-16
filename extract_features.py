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

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()
 
def make_model():
    model=models.vgg16(pretrained=True).features[:14]
    # model=model.eval()
    model.cuda()
    return model

def extract_feature(model,imgpath):
    model.eval()

    img=Image.open(imgpath)	
    img=img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor=img_to_tensor(img)
    tensor=tensor.resize_(1,3,TARGET_IMG_SIZE,TARGET_IMG_SIZE)
    tensor=tensor.cuda()
    
    result=model(Variable(tensor))
    result_npy=result.data.cpu().numpy()
    
    return result_npy[0]
    
if __name__=="__main__":
    imgpath='/bmrNAS/people/yuxinh/DL_diffseg/vgg_features'
    subjects = sorted(glob(os.path.join(imgpath, 'im*')))
    # print(subjects)
    idx = 1
    for subject_id in subjects:
        # subject_id = subjects[0]
        model=make_model()
        tmp = extract_feature(model, os.path.join(imgpath, subject_id))
        print(tmp.shape)
        sio.savemat(os.path.join(imgpath, "features_") + str(idx) + ".mat", \
          mdict={'features': tmp})
        idx += 1
    # print(tmp)