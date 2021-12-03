import torch
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import cv2
import torch.nn.init as int


# -------------Initialization----------------------------------------
# def init_weights(*modules):
#     for module in modules:
#         for m in module.modules():
#             if isinstance(m, nn.Conv2d):
#                 # try:
#                 #     import tensorflow as tf
#                 #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
#                 #     m.weight.data = tensor.eval()
#                 # except:
#                 #     print("try error, run variance_scaling_initializer")
#                 # variance_scaling_initializer(m.weight)
#                 # variance_scaling_initializer(m.weight)
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.Linear):
#                 # variance_scaling_initializer(m.weight)
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
# -------------Initialization----------------------------------------
def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def summaries(model, writer=None, grad=False, torchsummary=None):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs
class panbranch(nn.Module):
    def __init__(self, conv=default_conv):
        super(panbranch, self).__init__()
        channel = 8
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1,
                                    bias=True)
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv_output1 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1,
                                      bias=True)
        self.downscale = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.conv_22 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        init_weights(self.conv_input, self.conv_1, self.conv_2, self.conv_output1, self.downscale, self.conv_12,
                     self.conv_22)

    def forward(self, pan):  # h*w*1
        feature_1 = self.conv_input(pan)  # h*w*64
        feature_2 = self.relu(self.conv_1(feature_1))  # h*w*64
        feature_3 = self.conv_2(feature_2)  # h*w*64
        detail_out = self.relu(self.conv_output1(torch.add(feature_3, feature_1)))  # h*w*8

        feature_1_down = self.downscale(torch.add(feature_3, feature_1))
        feature_2_down = self.relu(self.conv_12(feature_1_down))  # h*w*64
        feature_3_down = self.conv_22(feature_2_down)  # h*w*64
        detail_out_down = self.relu(self.conv_output1(torch.add(feature_3_down, feature_1_down)))  # h*w*64
        return detail_out, detail_out_down
class mutiscale_net(nn.Module):
    def __init__(self):
        super(mutiscale_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=60, kernel_size=7, stride=1, padding=3,
                               bias=True)
        self.conv2_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(in_channels=30, out_channels=8, kernel_size=5, stride=1, padding=2,
                               bias=True)
        self.shallow1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=9, stride=1, padding=4,
                                  bias=True)
        self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.shallow3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.relu = nn.ReLU(inplace=True)
        init_weights(self.conv1, self.conv2_1, self.conv2_2, self.conv2_3, self.conv3, self.conv4_1, self.conv4_2,
                     self.conv4_3, self.conv5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out21 = self.conv2_1(out1)
        out22 = self.conv2_2(out1)
        out23 = self.conv2_3(out1)
        out2 = torch.cat([out21, out22, out23], 1)
        out2 = self.relu(torch.add(out2, out1))
        out3 = self.relu(self.conv3(out2))
        out41 = self.conv4_1(out3)
        out42 = self.conv4_2(out3)
        out43 = self.conv4_3(out3)
        out4 = torch.cat([out41, out42, out43], 1)
        out4 = self.relu(torch.add(out4, out3))
        out5 = self.conv5(out4)
        shallow1 = self.relu(self.shallow1(x))
        shallow2 = self.relu(self.shallow2(shallow1))
        shallow3 = self.shallow3(shallow2)
        out = torch.add(out5, shallow3)
        return out
class MRAB(nn.Module):
    def __init__(self, conv=default_conv):
        super(MRAB, self).__init__()
        channel = 8
        self.pa = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, lms, detail):
        lms = torch.tensor(lms, dtype=torch.float32).cuda()
        w = self.pa(torch.cat([lms, detail], dim=1)).cuda()  # dm??i
        x = w * detail
        out = torch.add(lms, x)

        return out


class TDNet(nn.Module):
    def __init__(self):
        super(TDNet, self).__init__()

        spectral_num = 8
        upscale = 2
        self.pan_branch = panbranch()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv_up = nn.Conv2d(in_channels=spectral_num, out_channels=spectral_num * upscale * upscale, kernel_size=3,
                                 stride=1, padding=1, bias=True)
        self.subpixel_up = nn.PixelShuffle(upscale)

        self.downscale_pan = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=spectral_num, out_channels=spectral_num * upscale * upscale,
                      kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.Dropout(0.1)
        )
        self.subpixel_up2 = nn.PixelShuffle(upscale)
        self.mulnet = mutiscale_net()
        self.mulnet2 = mutiscale_net()
        self.mrab = MRAB()
        self.mrab2 = MRAB()

        init_weights(self.mulnet, self.mulnet2, self.conv_up, self.subpixel_up, self.conv_up2, self.subpixel_up2,
                     self.mrab2, self.mrab, self.downscale_pan)

    def forward(self, x, y):  # x= ms; y =pan ; z=pan(downsample)
        # pan branch

        # global rs, rs1
        num_block = 6
        detail_out, detail_out_down = self.pan_branch(y)
        ms_up_1 = self.conv_up(x)
        ms_up_1 = self.subpixel_up(ms_up_1)
        ms_up_1 = self.mrab(ms_up_1, detail_out_down)
        ms_up_1_out = self.mulnet(torch.cat([ms_up_1, detail_out_down], dim=1))

        ms_up_2 = self.conv_up2(ms_up_1_out)
        ms_up_2 = self.subpixel_up2(ms_up_2)
        ms_up_2 = self.mrab2(ms_up_2, detail_out)
        ms_up_2_out = self.mulnet2(torch.cat([ms_up_2, detail_out], dim=1))

        return ms_up_1_out, ms_up_2_out