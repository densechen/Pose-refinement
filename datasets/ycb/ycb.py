'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:44:03
'''
import os
import random

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes

import utils
from datasets.base_dataset import BaseDataset


class YCBDataset(BaseDataset):
    def __init__(self, settings, mode="train", obj_class=None):
        super().__init__(settings)
        self.mode = mode
        # 1. FILELIST
        with open(self.settings.FILELIST[mode], "r") as f:
            self.list = [line.replace("\n", "") for line in f]

        # 2. MESH
        meshlist = [
            os.path.join(self.settings.DATA_ROOT, "models", m, "textured.obj")
            for m in self.settings.MODEL_DICT.keys()
        ]
        self.mesh = load_objs_as_meshes(meshlist)

        self.obj_class = obj_class
        # 3. POINT USED FOR ICP
        pointlist = [
            os.path.join(self.settings.DATA_ROOT, "models", m, "points.xyz")
            for m in self.settings.MODEL_DICT.keys()
        ]
        point = [
            torch.from_numpy(np.loadtxt(
                f, dtype=float)).float()[:settings.NUM_POINTS]
            for f in pointlist
        ]
        self.point = point

    def __len__(self):
        return len(self.list)

    def load_data(self, index):
        frame_path = os.path.join(self.settings.DATA_ROOT, self.list[index])
        meta = scio.loadmat(frame_path + "-meta.mat")
        obj = np.array(meta["cls_indexes"], dtype=int).flatten().tolist()
        label = np.array(Image.open(frame_path + "-label.png"))
        depth = np.array(Image.open(frame_path + "-depth.png"),
                         dtype=np.float32) / float(meta["factor_depth"][0, 0])
        color = np.array(
            Image.open(frame_path + "-color.png"))[..., :3] / 255.0

        label = np.fliplr(label)
        depth = np.fliplr(depth)
        color = np.fliplr(color)

        if self.obj_class is not None:
            if self.obj_class in obj:
                class_index = obj.index(self.obj_class)
                class_id = self.obj_class
            else:
                return self.load_data(random.choice(range(len(self))))
        else:
            class_index = random.choice(range(len(obj)))
            class_id = obj[class_index]

        # OBSERVED
        mask_depth = 1 - (depth == 0)
        mask_label = label == class_id
        mask = mask_label * mask_depth

        if self.mode == "train":
            mask = self.dilated_mask(mask)  # DILATE MASK
        if np.sum(mask) < 2500:
            return self.load_data(random.choice(range(len(self))))

        mask = np.expand_dims(mask, axis=-1)
        depth = np.expand_dims(depth, axis=-1)
        color = color * mask
        observed = torch.from_numpy(color).float()
        depth = depth * mask

        # POSE
        rotation = torch.from_numpy(
            np.array(meta["poses"][:3, :3, class_index].T,
                     dtype=np.float32)).view(1, 3, 3)
        translation = torch.from_numpy(
            np.array(meta["poses"][:3, 3, class_index],
                     dtype=np.float32)).view(1, 3)
        pose = utils.Pose(Rotation=utils.build_rotation(rotation,
                                                        format="matrix"),
                          Translation=utils.build_translation(translation))
        pose = utils.squeeze_namedtuple(pose, dim=0)

        # K
        K = np.array(meta["intrinsic_matrix"], dtype=np.float32)
        intrinsic = utils.Intrinsic(fx=K[0, 0],
                                    fy=K[1, 1],
                                    cx=K[0, 2],
                                    cy=K[1, 2])

        # MESH
        mesh = self.mesh[class_id - 1]

        return mesh, observed, depth, intrinsic, mask, pose, self.point[
            class_id - 1]
