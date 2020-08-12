'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:42:56
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes

import utils

from .define_camera import define_camera


class DRender(nn.Module):
    def __init__(self, settings, device):
        super().__init__()

        self.settings = settings
        self.device = device

    def forward(self, pose: utils.Pose, mesh: Meshes, **kwargs):
        settings = kwargs.get("settings", self.settings)
        if settings.DIRECT_CROP:
            rendered_center_points = kwargs.get("rendered_center_points", None)
            if rendered_center_points is None:
                raise "rendered_center_points must be provided"
            cx, cy = settings.CX, settings.CY
            rendered_images, rendered_depths, rendered_masks = [], [], []
            for i, center in enumerate(rendered_center_points):
                x0, y0 = center[0] - settings.CROP_SIZE[0] // 2, center[
                    1] - settings.CROP_SIZE[1] // 2

                # if settings.DATASET == "ycb":
                #     x0 = settings.IMAGE_WIDTH - x0
                new_cx, new_cy = cx - x0, cy - y0
                phong_renderer, silhouete_renderer = define_camera(
                    max(settings.CROP_SIZE),
                    settings.CROP_SIZE[1],
                    settings.CROP_SIZE[0],
                    fx=settings.FX,
                    fy=settings.FY,
                    cx=new_cx,
                    cy=new_cy,
                    device=self.device)

                rendered_image, rendered_depth = phong_renderer(
                    meshes_world=mesh[i:i + 1],
                    R=pose.Rotation.matrix[i:i + 1],
                    T=pose.Translation.translation[i:i + 1])
                silhouete_image = silhouete_renderer(
                    meshes_world=mesh[i:i + 1],
                    R=pose.Rotation.matrix[i:i + 1],
                    T=pose.Translation.translation[i:i + 1],
                )

                rendered_images.append(rendered_image[..., :3])
                rendered_depths.append(rendered_depth[..., :1])
                rendered_masks.append(silhouete_image[..., -1:])

            def crop_data(data):
                return data[:, :settings.CROP_SIZE[1], :settings.CROP_SIZE[0]]

            return crop_data(torch.cat(rendered_images, dim=0)), \
                crop_data(torch.cat(rendered_depths,dim=0)), \
                              crop_data(torch.cat(rendered_masks, dim=0))

        else:
            phong_renderer, silhouete_renderer = define_camera(
                settings.IMAGE_SIZE, settings.IMAGE_HEIGHT,
                settings.IMAGE_WIDTH, settings.FX, settings.FY, settings.CX,
                settings.CY, self.device)

            # RENDER
            rendered_image, rendered_depth = phong_renderer(
                meshes_world=mesh,
                R=pose.Rotation.matrix,
                T=pose.Translation.translation,
            )
            silhouete_image = silhouete_renderer(
                meshes_world=mesh,
                R=pose.Rotation.matrix,
                T=pose.Translation.translation,
            )

            def crop_data(data):
                return data[:, :settings.IMAGE_HEIGHT, :settings.IMAGE_WIDTH]

            return crop_data(rendered_image[..., :3]), crop_data(
                rendered_depth[..., :1]), crop_data(silhouete_image[..., -1:])
