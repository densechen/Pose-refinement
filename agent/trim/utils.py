'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:43:32
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def get_bbox(mask):
    try:
        mask = mask.type(torch.bool)
        rows = torch.any(mask, dim=1)
        rows_index = torch.where(rows)[0]
        rmin, rmax = rows_index[0], rows_index[-1]

        cols = torch.any(mask, dim=0)
        cols_index = torch.where(cols)[0]
        cmin, cmax = cols_index[0], cols_index[-1]

        return utils.BBOX(left=cmin, right=cmax, low=rmin, top=rmax)
    except Exception as e:
        raise ZeroDivisionError("Too Small BBOX")


def crop_image(image, bbox: utils.BBOX, image_height, image_width):
    # PAD FIRST THEN CROP
    pad_left = -min(bbox.left, 0)
    pad_right = -min(image_width - bbox.right, 0)
    pad_low = -min(bbox.low, 0)
    pad_top = -min(image_height - bbox.top, 0)

    # image_pad = F.pad(image, pad=[pad_top, pad_low, pad_left, pad_right])
    image_pad = F.pad(image, pad=[pad_left, pad_right, pad_low, pad_top])

    left = bbox.left + pad_left
    right = bbox.right + pad_left
    top = bbox.top + pad_low
    low = bbox.low + pad_low

    return image_pad[..., low:top, left:right]


def aspect_bbox(obbox: utils.BBOX,
                rbbox: utils.BBOX,
                rcenter,
                ccenter,
                aspect_ratio,
                lamda=1.4):
    xdist = max([
        abs(obbox.left - ccenter),
        abs(rbbox.left - ccenter),
        abs(obbox.right - ccenter),
        abs(rbbox.right - ccenter)
    ])

    ydist = max([
        abs(obbox.top - rcenter),
        abs(rbbox.top - rcenter),
        abs(obbox.low - rcenter),
        abs(rbbox.low - rcenter)
    ])

    width = max(xdist, ydist * aspect_ratio) * lamda
    height = max(xdist / aspect_ratio, ydist) * lamda

    # NEW BBOX
    bbox = utils.BBOX(left=int((ccenter - width).item()),
                      right=int((ccenter + width).item()),
                      low=int((rcenter - height).item()),
                      top=int((rcenter + height).item()))
    return bbox
