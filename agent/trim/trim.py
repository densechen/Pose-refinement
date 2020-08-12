'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:43:28
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import utils

from .utils import aspect_bbox, crop_image, get_bbox


class Trim(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.settings = settings

        self.register_buffer("image_std",
                             torch.tensor(settings.IMAGE_STD).view(3, 1, 1))
        self.register_buffer("image_mean",
                             torch.tensor(settings.IMAGE_MEAN).view(3, 1, 1))

    def trim(self, observed_image, observed_depth, observed_mask,
             rendered_image, rendered_depth, rendered_mask,
             rendered_center_points):
        def clip_data(image):
            # channel first
            image = image.permute(2, 0, 1)
            return crop_image(image, bbox, self.settings.IMAGE_HEIGHT,
                              self.settings.IMAGE_WIDTH)

        if self.settings.DATASET == "ycb":
            rendered_center_points[
                0] = self.settings.IMAGE_WIDTH - rendered_center_points[0]

        if self.settings.DIRECT_CROP:
            bbox = utils.BBOX(left=int(rendered_center_points[0] -
                                       self.settings.CROP_SIZE[0] // 2),
                              low=int(rendered_center_points[1] -
                                      self.settings.CROP_SIZE[1] // 2),
                              right=int(rendered_center_points[0] +
                                        self.settings.CROP_SIZE[0] // 2),
                              top=int(rendered_center_points[1] +
                                      self.settings.CROP_SIZE[1] // 2))
            ratio = torch.tensor([
                (bbox.right - bbox.left) / self.settings.IMAGE_WIDTH
            ])
            return clip_data(observed_image), clip_data(
                observed_depth), clip_data(
                    observed_mask), rendered_image.permute(
                        2, 0, 1), rendered_depth.permute(
                            2, 0, 1), rendered_mask.permute(2, 0, 1), ratio

        else:
            ccenter, rcenter = rendered_center_points[
                0], rendered_center_points[1]

            obbox = get_bbox(observed_mask)
            rbbox = get_bbox(rendered_mask)

            bbox = aspect_bbox(obbox,
                               rbbox,
                               rcenter,
                               ccenter,
                               aspect_ratio=self.settings.ASPECT_RATIO,
                               lamda=self.settings.MASK_ENLARGE)

            ratio = torch.tensor([
                (bbox.right - bbox.left) / self.settings.IMAGE_WIDTH
            ])

            return clip_data(observed_image), clip_data(
                observed_depth), clip_data(observed_mask), clip_data(
                    rendered_image), clip_data(rendered_depth), clip_data(
                        rendered_mask), ratio

    def forward(self, observed_image, observed_depth, observed_mask,
                rendered_image, rendered_depth, rendered_mask,
                rendered_center_points, rendered_center_depth, **kwargs):
        size = [self.settings.IMAGE_HEIGHT, self.settings.IMAGE_WIDTH]
        trim_observed_image, trim_observed_depth, trim_observed_mask = [], [], []
        trim_rendered_image, trim_rendered_depth, trim_rendered_mask = [], [], []
        interpolate_ratio = []
        for o_image, o_depth, o_mask, r_image, r_depth, r_mask, r_points, r_cdepth in zip(
                observed_image, observed_depth, observed_mask, rendered_image,
                rendered_depth, rendered_mask, rendered_center_points,
                rendered_center_depth):
            to_image, to_depth, to_mask, tr_image, tr_depth, tr_mask, ratio = self.trim(
                o_image, o_depth, o_mask, r_image, r_depth, r_mask, r_points)

            # NORM DEPTH
            # tr_depth = tr_depth - r_cdepth
            # tr_depth[tr_mask < 0.5] = 0.0

            # to_depth = to_depth - r_cdepth
            # to_depth[to_mask < 0.5] = 0.0
            tr_mask = (tr_mask > 0).float()

            tr_depth = tr_depth * tr_mask
            to_depth = to_depth * to_mask

            # NORM COLOR
            # tr_image = (tr_image - self.image_mean) / self.image_std
            # to_image = (to_image - self.image_mean) / self.image_std

            # tr_tmp = tr_mask.expand_as(tr_image) < 0.5
            # tr_image[tr_tmp] = 0.0
            # to_tmp = to_mask.expand_as(to_image) < 0.5
            # to_image[to_tmp] = 0.0

            tr_image = tr_image * tr_mask
            to_image = to_image * to_mask

            # TRANSPOSE AND SCALE
            ts = lambda x: F.interpolate(x.float().unsqueeze(dim=0), size=size)
            to_image = ts(to_image)
            to_depth = ts(to_depth)
            to_mask = ts(to_mask)

            tr_image = ts(tr_image)
            tr_depth = ts(tr_depth)
            tr_mask = ts(tr_mask)

            trim_observed_image.append(to_image)
            trim_observed_depth.append(to_depth)  # DEPTH NORM
            trim_observed_mask.append(to_mask)

            trim_rendered_image.append(tr_image)
            trim_rendered_depth.append(tr_depth)  # DEPTH NORM
            trim_rendered_mask.append(tr_mask)

            interpolate_ratio.append(ratio)

        if self.settings.DEBUG:
            time = kwargs.get("time", 1)
            utils.save_image_pair(
                tr_image[0].permute(1, 2, 0), to_image[0].permute(1, 2, 0),
                "{}/trim_rendered_{}.png".format(self.settings.DEBUG_PATH,
                                                 time),
                "{}/trim_observed_{}.png".format(self.settings.DEBUG_PATH,
                                                 time),
                "{}/trim_rendered_observed_{}.png".format(
                    self.settings.DEBUG_PATH, time))
            utils.save_image_pair(
                tr_mask[0].permute(1, 2, 0).repeat(1, 1, 3),
                to_mask[0].permute(1, 2, 0).repeat(1, 1, 3),
                "{}/trim_rendered_mask_{}.png".format(self.settings.DEBUG_PATH,
                                                      time),
                "{}/trim_observed_mask_{}.png".format(self.settings.DEBUG_PATH,
                                                      time),
                "{}/trim_rendered_observed_mask_{}.png".format(
                    self.settings.DEBUG_PATH, time))

        trim_observed_image = torch.cat(trim_observed_image, dim=0)
        trim_observed_depth = torch.cat(trim_observed_depth, dim=0)
        trim_observed_mask = torch.cat(trim_observed_mask, dim=0)

        trim_rendered_image = torch.cat(trim_rendered_image, dim=0)
        trim_rendered_depth = torch.cat(trim_rendered_depth, dim=0)
        trim_rendered_mask = torch.cat(trim_rendered_mask, dim=0)

        interpolate_ratio = torch.stack(interpolate_ratio,
                                        dim=0).type_as(trim_rendered_image)

        return trim_observed_image, trim_observed_depth, trim_observed_mask, trim_rendered_image, trim_rendered_depth, trim_rendered_mask, interpolate_ratio
