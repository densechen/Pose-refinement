'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:32
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from trainer.loss import flow_loss, mask_loss, point_loss

__all__ = ["trainer"]


def trainer(agent, mesh, raw_data, optimizer, settings):
    """ zoom_observed_image, zoom_observed_depth, zoom_observed_mask,
        zoom_rendered_image, zoom_rendered_depth, zoom_rendered_mask
    """
    source_pose = raw_data.init_pose
    intrinsic = raw_data.Intrinsic

    settings.set_intrinsic(intrinsic)

    optimizer.zero_grad()
    losses = []
    reward_losses = []
    for _ in range(settings.EPISODE_LEN):
        if settings.is_synthetic_dataset():
            source_pose = utils.detach_namedtuple(source_pose)
            data = raw_data.data
            interpolate_ratio = raw_data.interpolate_ratio
        else:
            center_points, center_depth = utils.translation_to_voxel_and_depth(
                source_pose.Translation.translation, intrinsic, settings)
            data, interpolate_ratio = agent.synthetic(
                observed_image=raw_data.image,
                observed_depth=raw_data.depth,
                observed_mask=raw_data.mask,
                init_pose=source_pose,
                mesh=mesh,
                center_points=center_points,
                center_depth=center_depth,
                settings=settings)
        state_feature, mask, flow = agent.state_encoding(data)
        action = agent.action_encoding(state_feature, interpolate_ratio)

        last_pose = source_pose
        source_pose = utils.apply_action_to_pose(action, source_pose, settings)
        if settings.WITH_REWARD:
            reward = agent.reward_encoding(state_feature, action)
            expect_reward = utils.calculate_expected_reward(
                last_pose, source_pose, raw_data.target_pose)
            reward_loss = F.l1_loss(reward, expect_reward.view_as(reward))
            reward_loss.backward(retain_graph=True)
            reward_losses.append(reward_loss.item())
        else:
            reward_losses.append(0.0)

        loss = 0.0
        if settings.MASK_LOSS:
            loss += mask_loss(
                mask, data[:, -1:].detach()) * settings.MASK_LOSS_WEIGHT
        if settings.FLOW_LOSS:
            loss += flow_loss(predict_flow=flow,
                              observed_depth=data[:, 1:2],
                              rendered_depth=data[:, -2:-1],
                              source_pose=utils.detach_namedtuple(source_pose),
                              target_pose=raw_data.target_pose,
                              settings=settings) * settings.FLOW_LOSS_WEIGHT
        if settings.POINT_WISE_LOSS:
            loss += point_loss(
                raw_data.model_points, source_pose,
                raw_data.target_pose) * settings.POINT_WISE_LOSS_WEIGHT
        loss.backward(retain_graph=True)
        losses.append(loss.item())

    optimizer.step()

    return sum(losses) / len(losses), sum(reward_losses) / len(reward_losses)
