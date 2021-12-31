import os
import glob 
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from . import hyperparameters as params

def weights_init(m, init_gain=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        #torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_l16 = nn.Linear(256, 256)
        self.fc_l12 = nn.Linear(256, 256)
        self.fc_l8 = nn.Linear(256, 256)
        self.fc_l4 = nn.Linear(128, 256)
        self.fc_l0 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.l2norm = Normalize(2)
  
    def forward(self, map, sample_list=None, patch_num=256, enc_layer=0):
        B, _, H, W = map.shape
        feat_reshape = map.permute(0, 2, 3, 1).flatten(1, 2) # [B, S=H*W, C]
        #print(feat_reshape.size())
        #print(feat_reshape)
        N = min(H*W, patch_num)
        if sample_list is None:
          sample_list = random.sample(range(H*W), N)
        #print(sample_list) # len=N
        x_sample = feat_reshape[:, [sample_list], :].flatten(0, 1) #[B * N, C]
        
        #print(x_sample.size())
        #print(x_sample)
              
        if enc_layer == 16:
          out = F.relu(self.fc_l16(x_sample))
        elif enc_layer == 12:
          out = F.relu(self.fc_l12(x_sample))
        elif enc_layer == 8:
          out = F.relu(self.fc_l8(x_sample))
        elif enc_layer == 4:
          out = F.relu(self.fc_l4(x_sample))
        else:
          out = F.relu(self.fc_l0(x_sample))
        out = self.fc2(out)
        out = self.l2norm(out)
        out = torch.reshape(out, (B, N, -1))
        out = torch.transpose(out, 1, 2)

        """
        out = torch.unsqueeze(out, -1)

        if idx == 0:
          stacked_out = out
        else:
          stacked_out = torch.cat((stacked_out, out), dim=-1)
        """
        return out, sample_list

class Discriminator(nn.Module):
    def __init__(self, input_shape=(params.nc, params.image_size, params.image_size)):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class Generator(nn.Module):
    def __init__(self, input_shape=(params.nc, params.image_size, params.image_size), num_residual_blocks=9):
        super(Generator, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, enc_layer=None):
        if enc_layer is not None:
          feat = x
          enc_feats = []
          for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id in enc_layer:
              enc_feats.append(feat)
            if layer_id == enc_layer[-1]:
              return enc_feats
        return self.model(x)


# Input: f_q (BxCxS) and sampled features from H(G_enc(x))
# Input: f_k (BxCxS) are sampled features from H(G_enc(G(x))
# Input: tau is the temperature used in PatchNCE loss.
# Output: PatchNCE loss
def PatchNCELoss(f_q, f_k, device, tau=0.07):
    # print(f_q.shape)
    # print(f_k.shape)
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    # batch size, channel size, and number of sample locations
    B, C, S = f_q.shape
    f_k = f_k.detach()

    # calculate v * v+: BxSx1
    l_pos = (f_k * f_q).sum(dim=1)[:, :, None]

    # calculate v * v-: BxSxS
    l_neg = torch.bmm(f_q.transpose(1, 2), f_k)

    # The diagonal entries are not negatives. Remove them.
    identity_matrix = torch.eye(S, dtype=torch.bool)[None, :, :].to(device)
    l_neg.masked_fill_(identity_matrix, -float('inf'))

    # calculate logits: (B)x(S)x(S+1)
    logits = torch.cat((l_pos, l_neg), dim=2) / tau

    # return PatchNCE loss
    predictions = logits.flatten(0, 1)
    targets = torch.zeros(B * S, dtype=torch.long).to(device)
    return cross_entropy_loss(predictions, targets)
