'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:26
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from .utils import (get_reward, train_actor_critic, train_discrim)

__all__ = ["irl_trainer"]


def irl_trainer(agent, mesh, raw_data, demostrations, optimizer, settings):
    # ENCODE RAW DATA INTO STATE_FEATURE
    source_pose = raw_data.init_pose
    intrinsic = raw_data.Intrinsic

    settings.set_intrinsic(intrinsic)

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
    state_features, _, _ = agent.state_encoding(data)
    actions = agent.action_encoding(state_features, interpolate_ratio)

    # train vdb
    train_discrim(agent.vdb, state_features, actions, optimizer, demostrations,
                  settings)

    # train actor critic
    rewards = get_reward(agent.vdb, state_features, actions)

    masks = [
        i % settings.SYNTHETIC_EPISODE_LEN != 0
        for i in range(1,
                       len(state_features) + 1)
    ]
    masks = torch.tensor(masks).to(settings.DEVICE)
    loss = train_actor_critic(agent.actor, agent.critic, state_features,
                              interpolate_ratio, actions, rewards, masks,
                              optimizer, settings)

    return loss