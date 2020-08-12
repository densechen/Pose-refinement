'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:44:12
'''
import os

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes

import utils
from datasets.base_dataset import BaseDataset


class YCBTestDataset(BaseDataset):
    """
    This class will load the predict result of PoseCNN.
    Will return all the data need for eval.
    """
    def __init__(self, settings, mode="test", obj_class=None):
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
            torch.from_numpy(np.loadtxt(f, dtype=float))[:settings.NUM_POINTS]
            for f in pointlist
        ]
        self.point = point

    def __len__(self):
        return len(self.list)

    def load_data(self, index):
        # yield the data in a frame
        frame_path = os.path.join(self.settings.DATA_ROOT, self.list[index])
        posecnn_meta = scio.loadmat("{}/results_PoseCNN_RSS2018/{}.mat".format(
            self.settings.YCB_TOOLBOX_DIR, "{:06d}".format(index)))
        label = np.array(posecnn_meta["labels"])
        posecnn_rois = np.array(posecnn_meta["rois"])
        posecnn_pose = np.array(posecnn_meta["poses"])

        depth = np.array(Image.open(frame_path + "-depth.png"),
                         dtype=np.float32) / 10000.0
        color = np.array(
            Image.open(frame_path + "-color.png"))[..., :3] / 255.0

        label = np.fliplr(label)
        depth = np.fliplr(depth)
        color = np.fliplr(color)

        lst = posecnn_rois[:, 1:2].flatten()

        for idx in range(len(lst)):
            itemid = int(lst[idx])

            # POSE
            rotation = torch.from_numpy(
                np.array(posecnn_pose[idx, :4], dtype=np.float32)).view(1, 4)
            translation = torch.from_numpy(
                np.array(posecnn_pose[idx, 4:], dtype=np.float32)).view(1, 3)
            rotation = utils.build_rotation(rotation, format="quat")

            pose = utils.Pose(Rotation=utils.build_rotation(
                rotation.matrix.permute(0, 2, 1), format="matrix"),
                              Translation=utils.build_translation(translation))
            pose = utils.squeeze_namedtuple(pose, dim=0)

            # mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
            # mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, itemid))
            # mask = mask_label * mask_depth
            mask_depth = 1 - (depth == 0)
            mask_label = label == itemid
            mask = mask_label * mask_depth

            mask = np.expand_dims(mask, axis=-1)
            depth_ = np.expand_dims(depth, axis=-1)
            color_ = color * mask
            observed = torch.from_numpy(color_).float()
            depth_ = depth_ * mask

            # K
            intrinsic = utils.Intrinsic(fx=1066.778,
                                        fy=1067.487,
                                        cx=312.9869,
                                        cy=241.3109)

            # MESH
            mesh = self.mesh[itemid - 1]

            data = utils.TestData(image=observed,
                                  depth=depth_,
                                  mask=mask,
                                  Intrinsic=intrinsic,
                                  init_pose=pose,
                                  model_points=self.point[itemid - 1])

            yield {"mesh": mesh, "data": data}
