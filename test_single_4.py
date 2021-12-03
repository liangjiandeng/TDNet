import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from model_band4 import BDPN_mra_band4
import h5py
import scipy.io as sio
import os
import torch.nn.functional as F

def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs

def load_gt_compared(file_path_gt,file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    gt = torch.from_numpy(data1['gt']/2047)
    compared_data = torch.from_numpy(data2['output_dmdnet_newdata6']*2047)
    return gt, compared_data

def load_gt_compared_band4(file_path_gt,file_path_compared):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    gt = torch.from_numpy(data1['gt']/2047)
    #compared_data = torch.from_numpy(data2['output_dmdnet_GF_data2']*2047)output_dmdnet_QB_data2
    compared_data = torch.from_numpy(data2['output_dmdet'] * 2047)
    return gt, compared_data
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / 2047).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047)   # HxW = 256x256
    return lms, ms, pan
ckpt = "BAND_4/390.pth"#bdpn_mra11

#ckpt = "find_best/(LR=0.001)220+(LR=0.0001)120+(LR=0.00005)50.pth"   # chose model

def test(file_path):
    lms, ms, pan = load_set(file_path)

    model = BDPN_mra_band4().cuda().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight)
    #model = torch.load(ckpt)["model"]
    with torch.no_grad():

        x1, x2, x3 = lms, ms, pan   # read data: CxHxW (numpy type)
        print(x1.shape)
        x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x2 = x2.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x3 = x3.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
        #pan_down = F.interpolate(x3, scale_factor=0.5).cuda()
        out1, out2 = model(x2, x3)  # tensor type: CxHxW
       # tensor type: CxHxW

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(out2).permute(1, 2, 0).cpu().detach().numpy()

        print(sr.shape)
        save_name = os.path.join("results", "output_zhang_GF_data2.mat")
        sio.savemat(save_name, {'output_zhang_GF_data2': sr})


if __name__ == '__main__':
    file_path = "test_data/GF_data2.mat"
    test(file_path)
