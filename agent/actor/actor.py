'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:42:32
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.settings = settings

        self.fc1 = nn.Linear(settings.STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 256)

        self.rot_1 = nn.Linear(256, 128)
        self.trans_1 = nn.Linear(256, 128)

        self.rot_2 = nn.Linear(128, settings.ROT_DIM)
        self.trans_2 = nn.Linear(128, settings.TRANS_DIM)

    def forward(self, x, ratio=1.0):
        """NOTE: If we interpolate input image, we should pay attention to the translation changing by this operation.
        """
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        rot = F.leaky_relu(self.rot_1(x))
        rot = self.rot_2(rot)

        trans = F.leaky_relu(self.trans_1(x))
        # trans = self.trans_2(trans) * ratio
        trans = self.trans_2(trans)  # WITHOUT RATIO

        mu = torch.cat([rot, trans], dim=-1)

        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        return mu, std
