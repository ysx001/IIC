import os.path as osp
import numpy as np
import scipy.io as sio
from glob import glob
from sklearn.decomposition import PCA

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
    # root='/bmrNAS/people/yuxinh/DL_diffseg/DiffSeg-Data/'
    # subjects = sorted(glob(osp.join(root, 'mwu*')))

    root='/bmrNAS/people/yuxinh/DL_diffseg/MSSeg-Data/'
    subjects = sorted(glob(osp.join(root, '*')))

    # print(subjects)
    model = make_model()
    for subject_id in subjects:
        slices = sorted(glob(osp.join(root, subject_id, 'jpg', "im*")))
        print(slices)
        sub_features = np.zeros((256, 32, 32, len(slices)))
        for i in range(len(slices)):
            tmp = extract_feature(model, slices[i])
            sub_features[:, :, :, i] = tmp
            print(tmp.shape, sub_features.shape)
            print(tmp.min(), tmp.max(), sub_features.min(), sub_features.max())
        sio.savemat(osp.join(root, subject_id, "features.mat"), {
            'features': sub_features,
        })