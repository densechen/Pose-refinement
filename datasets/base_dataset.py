'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:44:20
'''
import random

import cv2
import numpy as np
import pytorch3d.transforms as transforms3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import iterative_closest_point

import utils


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, settings):
        self.settings = settings

    def dilated_mask(self, mask):
        # without batch
        # mask should be cv2 image or numpy array
        mask = np.array(mask * 255, dtype=np.uint8)
        if self.settings.DEBUG:
            cv2.imwrite(
                "{}/before_dilate.png".format(self.settings.DEBUG_PATH), mask)
        kernel = np.ones((self.settings.DILATED_KERNEL_SIZE,
                          self.settings.DILATED_KERNEL_SIZE), np.uint8)
        mask = cv2.dilate(mask, kernel)
        if self.settings.DEBUG:
            cv2.imwrite("{}/after_dilate.png".format(self.settings.DEBUG_PATH),
                        mask)
        return np.array(mask > 127, dtype=np.uint8)

    def add_noise_to_pose(self, pose: utils.Pose):
        # with batch
        # RANDOM POSE
        euler = torch.tensor([
            random.choice([1, -1]) * random.random(),
            random.choice([1, -1]) * random.random(),
            random.choice([1, -1]) * random.random()
        ]).view(1, 3) * self.settings.NOISE_ROT
        trans = torch.tensor([
            random.choice([1, -1]) * random.random(),
            random.choice([1, -1]) * random.random(),
            random.choice([1, -1]) * random.random()
        ]).view(1, 3) * self.settings.NOISE_TRANS

        delta_pose = utils.Pose(Rotation=utils.build_rotation(euler,
                                                              format="euler"),
                                Translation=utils.build_translation(trans))

        return utils.apply_transform_to_pose(pose, delta_pose)

    def multi_angle_icp(self, source_points, target_points):
        # build initial transform
        angles = torch.tensor([[i, j, k] for i in range(0, 360, 180)
                               for j in range(0, 360, 180)
                               for k in range(0, 360, 180)]).float() / 360.0
        batch_size = len(angles)
        T = torch.mean(target_points, dim=0).unsqueeze(0).repeat(batch_size, 1)
        init_transform = SimilarityTransform(R=utils.build_rotation(
            angles, format="euler").matrix,
                                             T=T,
                                             s=torch.ones(batch_size))

        source_points, target_points = source_points.unsqueeze(0).repeat(
            batch_size, 1,
            1), target_points.unsqueeze(0).repeat(batch_size, 1, 1)

        icp = iterative_closest_point(source_points,
                                      target_points,
                                      init_transform=init_transform,
                                      allow_reflection=True)

        index = torch.min(icp.rmse, dim=0)[1]

        RTs = icp.RTs

        return utils.Pose(utils.build_rotation(RTs.R[index].unsqueeze(0)),
                          utils.build_translation(RTs.T[index].unsqueeze(0)))

    def load_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        mesh, image, depth, intrinsic, mask, target_pose, model_points = self.load_data(
            index)

        if self.settings.INIT_POSE_METHOD == "NOISE":
            init_pose = self.add_noise_to_pose(
                utils.unsqueeze_namedtuple(target_pose, dim=0))
            init_pose = utils.squeeze_namedtuple(init_pose, dim=0)
        elif self.settings.INIT_POSE_METHOD == "ICP":
            raise NotImplementedError

        return {
            "mesh":
            mesh,
            "data":
            utils.RawData(image, depth, intrinsic, mask, target_pose,
                          init_pose, model_points)
        }
