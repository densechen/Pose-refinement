import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_gather, knn_points

import utils

from .flow import calc_flow


def point_loss(model_points, source_pose, target_pose, simi=False):
    source_point = utils.transform_point_cloud(model_points, source_pose)
    target_point = utils.transform_point_cloud(model_points, target_pose)

    if simi:
        distance = chamfer_distance(source_point,
                                    target_point,
                                    point_reduction="sum")[0]
    else:
        distance = F.mse_loss(source_point, target_point, reduction="sum")

    return distance


def mask_loss(predict_mask, ground_mask):
    predict_mask = F.interpolate(
        predict_mask, size=[ground_mask.size(-2),
                            ground_mask.size(-1)])

    return F.binary_cross_entropy_with_logits(predict_mask, ground_mask)


def flow_loss(predict_flow, observed_depth, rendered_depth, source_pose,
              target_pose, settings):
    flows, visibles = [], []
    predict_flow = F.interpolate(
        predict_flow, size=[observed_depth.size(-2),
                            observed_depth.size(-1)])
    for i in range(len(predict_flow)):
        o_depth, r_depth, s_rot, s_trans, t_rot, t_trans = observed_depth[
            i], rendered_depth[i], source_pose.Rotation.matrix[
                i], source_pose.Translation.translation[
                    i], target_pose.Rotation.matrix[
                        i], target_pose.Translation.translation[i]
        o_depth, r_depth = o_depth + t_trans[-1], r_depth + t_trans[
            -1]  # DENORMALIZE

        # Convert to numpy
        # Channel Last
        o_depth = o_depth.permute(1, 2, 0).squeeze(dim=-1).cpu().numpy()
        r_depth = r_depth.permute(1, 2, 0).squeeze(dim=-1).cpu().numpy()
        s_pose = torch.cat([s_rot, s_trans.view(3, 1)], dim=-1).cpu().numpy()
        t_pose = torch.cat([t_rot, t_trans.view(3, 1)], dim=-1).cpu().numpy()

        k = np.eye(3)
        k[0, 0] = settings.FX
        k[1, 1] = settings.FY
        k[0, 2] = settings.CX
        k[1, 2] = settings.CY

        flow, visible, _ = calc_flow(r_depth, s_pose, t_pose, k, o_depth)

        flow = torch.from_numpy(flow)
        visible = torch.from_numpy(visible)
        flows.append(flow)
        visibles.append(visible)
    flows = torch.stack(flows, dim=0).type_as(predict_flow).permute(0, 3, 1, 2)
    visibles = torch.stack(visibles,
                           dim=0).type_as(predict_flow).unsqueeze(dim=1)

    return F.mse_loss(predict_flow * visibles, flows * visibles)
