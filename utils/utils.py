'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:43
'''
import math
from collections import Counter, OrderedDict, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch3d.ops import iterative_closest_point
from pytorch3d.transforms import (euler_angles_to_matrix,
                                  matrix_to_euler_angles, matrix_to_quaternion,
                                  quaternion_to_matrix, random_rotations)
from torch.distributions import Normal

"""
DATA STRUCTURE BASED ON NAMEDTUPLE
"""
PI = math.pi
Intrinsic = namedtuple("Intrinsic", ["fx", "fy", "cx", "cy"])
Rotation = namedtuple("Rotation", ["euler", "quat", "matrix", "ortho6d"])
Translation = namedtuple("Translation", ["translation"])
Pose = namedtuple("Pose", ["Rotation", "Translation"])
RawData = namedtuple("RawData", [
    "image", "depth", "Intrinsic", "mask", "target_pose", "init_pose",
    "model_points"
])
TestData = namedtuple(
    "TestData",
    ["image", "depth", "Intrinsic", "mask", "init_pose", "model_points"])
SynRawData = namedtuple("SynRawData", [
    "data", "Intrinsic", "target_pose", "init_pose", "model_points",
    "interpolate_ratio"
])
Data = namedtuple("Data", [
    "observed_image", "observed_depth", "observed_mask", "rendered_image",
    "rendered_depth", "rendered_mask"
])
BBOX = namedtuple("BBOX", ["left", "right", "top", "low"])


def index_select_namedtuple(t, dim, index: torch.Tensor):
    return type(t)(*[
        torch.index_select(i, dim=dim, index=index) if isinstance(
            i, torch.Tensor) else index_select_namedtuple(i, dim, index)
        for i in t
    ])


def cat_namedtuple_list(t: list, dim=0):
    t_cat = type(t[0])(*[[] for _ in t[0]])
    for tt in t:
        for i, ttt in enumerate(tt):
            t_cat[i].append(ttt)
    t_catted = type(t[0])(*[
        torch.cat(i, dim=dim) if isinstance(i[0], torch.Tensor
                                            ) else cat_namedtuple_list(i, dim)
        for i in t_cat
    ])
    return t_catted


def stack_namedtuple_list(t: list, dim=0):
    t_stack = type(t[0])(*[[] for _ in t[0]])
    for tt in t:
        for i, ttt in enumerate(tt):
            t_stack[i].append(ttt)
    t_stacked = type(t[0])(*[
        torch.stack(i, dim=dim)
        if isinstance(i[0], torch.Tensor) else stack_namedtuple_list(i, dim)
        for i in t_stack
    ])
    return t_stacked


def repeat_namedtuple(t, shape: list):
    return type(t)(*[
        i.repeat(*shape) if isinstance(i, torch.Tensor
                                       ) else repeat_namedtuple(i, shape)
        for i in t
    ])


def squeeze_namedtuple(t, dim=0):
    return type(t)(*[
        i.squeeze(dim) if isinstance(i, torch.Tensor
                                     ) else squeeze_namedtuple(i, dim)
        for i in t
    ])


def unsqueeze_namedtuple(t, dim=0):
    return type(t)(*[
        i.unsqueeze(dim) if isinstance(i, torch.Tensor
                                       ) else unsqueeze_namedtuple(i, dim)
        for i in t
    ])


def variable_namedtuple(t, device):
    return type(t)(*[
        torch.autograd.Variable(i).to(device)
        if isinstance(i, torch.Tensor) else variable_namedtuple(i, device)
        for i in t
    ])


def detach_namedtuple(t):
    return type(t)(*[
        i.detach().clone() if isinstance(i, torch.Tensor
                                         ) else detach_namedtuple(i) for i in t
    ])


def numpy_namedtuple(t):
    return type(t)(*[
        i.detach().cpu().numpy(
        ) if isinstance(i, torch.Tensor) else numpy_namedtuple(i) for i in t
    ])


def append_namedtuple(t1, t2):
    for tt1, tt2 in zip(t1, t2):
        tt1.append(tt2)


def build_translation(translation) -> Translation:
    return Translation(translation)


def build_rotation(rotation, format="matrix") -> Rotation:
    """ Convert roation (with format) into Rotation.
    format: matrix, ortho6d, quat, euler
    """
    # 1. CONVERT SPECIFIED FORMAT TO MATRIX FIRST
    if format == "matrix":
        matrix = rotation
    elif format == "ortho6d":
        matrix = compute_rotation_matrix_from_ortho6d(rotation)
    elif format == "euler":
        matrix = euler_angles_to_matrix(rotation, convention="XYZ")
    elif format == "quat":
        matrix = quaternion_to_matrix(rotation)
    else:
        raise TypeError

    # 2. BUILD ROTATION
    return Rotation(
        ortho6d=rotation if format == "ortho6d" else
        compute_ortho6d_from_rotation_matrix(matrix),
        quat=rotation if format == "quat" else matrix_to_quaternion(matrix),
        matrix=rotation if format == "matrix" else matrix,
        euler=rotation if format == "euler" else matrix_to_euler_angles(
            matrix, convention="XYZ"))


def transpose_rotation(rotation: Rotation) -> Rotation:
    """ As for ortho6d and euler, we will first convert to matrix, then transpose.
    As others, we will use 
    """
    # 1. TRANSPOSE ORTHO6D
    matrix = compute_rotation_matrix_from_ortho6d(rotation.ortho6d).transpose(
        2, 1)
    ortho6d = compute_ortho6d_from_rotation_matrix(matrix)

    # 2. TRANSPOSE EULER
    matrix = euler_angles_to_matrix(rotation.euler,
                                    convention="XYZ").transpose(2, 1)
    euler = matrix_to_euler_angles(matrix, convention="XYZ")

    # 3. TRANSPOSE QUAT
    quat = quaternion_invert(rotation.quat)

    # 4. TRANSPOSE MATRIX
    matrix = rotation.matrix.transpose(2, 1)

    return Rotation(ortho6d=ortho6d, quat=quat, matrix=matrix, euler=euler)


def normalize_vector(v):
    n = torch.norm(v, dim=-1, keepdim=True)
    n = torch.clamp_min(n, min=1e-8)
    return v / n


def cross_vector(u, v):
    return torch.cross(u, v, dim=-1)


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, :3]
    y_raw = ortho6d[:, 3:]

    x = normalize_vector(x_raw)
    z = cross_vector(x, y_raw)
    z = normalize_vector(z)
    y = cross_vector(z, x)

    return torch.stack([x, y, z], dim=1)


def compute_ortho6d_from_rotation_matrix(rotation):
    x_raw = rotation[:, 0, :]
    y_raw = rotation[:, 1, :]
    return torch.cat([x_raw, y_raw], dim=-1)


def apply_action_to_pose(action, src_pose: Pose, settings) -> Pose:
    action_rotation = build_rotation(rotation=action[:, :settings.ROT_DIM],
                                     format=settings.ROT_FORMAT)
    action_delta_voxel = action[:, -settings.TRANS_DIM:]

    src_rotation = src_pose.Rotation
    src_translation = src_pose.Translation.translation

    # NEW
    z_tgt = src_translation[:, 2] / torch.exp(action_delta_voxel[:, 2])
    # y_tgt = (action_delta_voxel[:, 1] / settings.FY +
    #          src_translation[:, 1] / src_translation[:, 2]) * z_tgt
    # x_tgt = (action_delta_voxel[:, 0] / settings.FX +
    #          src_translation[:, 0] / src_translation[:, 2]) * z_tgt
    y_tgt = (action_delta_voxel[:, 1] +
             src_translation[:, 1] / src_translation[:, 2]) * z_tgt
    x_tgt = (action_delta_voxel[:, 0] +
             src_translation[:, 0] / src_translation[:, 2]) * z_tgt
    rotation = build_rotation(torch.bmm(src_rotation.matrix,
                                        action_rotation.matrix),
                              format="matrix")
    translation = build_translation(torch.stack([x_tgt, y_tgt, z_tgt], dim=-1))

    return Pose(Rotation=rotation, Translation=translation)
    # return Pose(Rotation=rotation,
    #             Translation=build_translation(src_translation))


# def apply_action_to_pose(action, src_pose: Pose, settings) -> Pose:
#     action_rotation = build_rotation(rotation=action[:, :settings.ROT_DIM],
#                                      format=settings.ROT_FORMAT)
#     action_translation = action[:, -settings.TRANS_DIM:]

#     src_rotation = src_pose.Rotation
#     src_translation = src_pose.Translation.translation

#     rotation = build_rotation(torch.bmm(src_rotation.matrix, action_rotation.matrix), format="matrix")
#     translation = build_translation(src_translation + action_translation)

#     return Pose(Rotation=rotation, Translation=translation)


def apply_transform_to_pose(pose: Pose, transform: Pose) -> Pose:
    transform_matrix = transform.Rotation.matrix
    transform_trans = transform.Translation.translation

    pose_matrix = pose.Rotation.matrix.expand_as(transform_matrix)
    pose_trans = pose.Translation.translation.expand_as(transform_trans)

    return Pose(Rotation=build_rotation(torch.bmm(pose_matrix,
                                                  transform_matrix),
                                        format="matrix"),
                Translation=build_translation(pose_trans + transform_trans))


def distance_normalization_function(x, k=3):
    return 1 - torch.exp(-x * k)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    # 1. MUL
    m = torch.bmm(m1, m2.transpose(1, 2))

    # 2. TR
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    # 3. CLAMP
    cos = torch.clamp(cos, min=-1, max=1)

    return torch.acos(cos)


def transform_point_cloud(points, pose: Pose):
    return torch.bmm(
        points,
        pose.Rotation.matrix) + pose.Translation.translation.unsqueeze(dim=1)


def translation_to_voxel_and_depth(translation, intrinsic, settings):
    pix_x = translation[:, 0] * intrinsic.fx / translation[:, 2] + intrinsic.cx
    pix_y = translation[:, 1] * intrinsic.fy / translation[:, 2] + intrinsic.cy

    # if settings.DATASET == "ycb":
    #     pix_x = settings.IMAGE_WIDTH - pix_x
    return torch.stack([pix_x, pix_y], dim=-1), translation[:, 2]


def blend_two_images(img1, img2):
    img = Image.blend(img1, img2, 0.5)
    return img


def check_bbox(bbox: BBOX, image_width, image_height):
    if isinstance(bbox.left, int):
        if bbox.left < 0 or bbox.right > image_width or bbox.low < 0 or bbox.top > image_height:
            return False
        else:
            return True
    else:
        if torch.any(bbox.left < 0) or torch.any(
                bbox.right > image_width) or torch.any(
                    bbox.low < 0) or torch.any(bbox.top > image_height):
            return False
        else:
            return True


def save_image_pair(image1, image2, filename1, filename2, pairname):
    image1 = np.asarray(image1.detach().cpu().numpy() * 255.0, dtype=np.uint8)
    image2 = np.asarray(image2.detach().cpu().numpy() * 255.0, dtype=np.uint8)
    image1 = Image.fromarray(image1).convert("RGBA")
    image2 = Image.fromarray(image2).convert("RGBA")
    image = blend_two_images(image1, image2)
    image.save(pairname)
    image1.save(filename1)
    image2.save(filename2)


def load_state_dict(model, pretrained_state_dict):
    model_dict = model.state_dict()

    # REMOVE USELESS PARAMETERS
    pretrained_state_dict = {
        k: v
        for k, v in pretrained_state_dict.items() if k in model_dict
    }

    # UPDATE STATE DICT
    model_dict.update(pretrained_state_dict)

    # LOAD
    model.load_state_dict(model_dict)


def calculate_expected_reward(last_pose: Pose, current_pose: Pose,
                              target_pose: Pose):
    last_distance = F.l1_loss(
        last_pose.Translation.translation, target_pose.Translation.translation
    ) + compute_geodesic_distance_from_two_matrices(
        last_pose.Rotation.matrix, target_pose.Rotation.matrix)
    current_distance = F.l1_loss(
        current_pose.Translation.translation, target_pose.Translation.
        translation) + compute_geodesic_distance_from_two_matrices(
            current_pose.Rotation.matrix, target_pose.Rotation.matrix)

    return (last_distance - current_distance).detach()  # if better, do it.
