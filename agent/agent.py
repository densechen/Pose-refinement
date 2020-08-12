'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:43:46
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes

import utils
from agent import actor, critic, drender, flownet, trim, vdb


class Agent(nn.Module):
    def __init__(self, settings, device):
        super().__init__()

        self.settings = settings
        self.drender = drender.DRender(settings, device)
        self.trim = trim.Trim(settings)
        self.flownet = flownet.__dict__[settings.ARCH](settings)
        self.actor = actor.Actor(settings)
        self.critic = critic.Critic(settings)
        self.vdb = vdb.VDB(settings)

    def synthetic(self, observed_image, observed_depth, observed_mask,
                  init_pose: utils.Pose, mesh: Meshes, center_points,
                  center_depth, **kwargs):
        settings = kwargs.get("settings", self.settings)

        if settings.DIRECT_CROP:
            rendered_image, rendered_depth, rendered_mask = self.drender(
                init_pose,
                mesh,
                settings=settings,
                rendered_center_points=center_points)
        else:
            rendered_image, rendered_depth, rendered_mask = self.drender(
                init_pose, mesh, settings=settings)

        # if self.settings.DEBUG:
        #     utils.save_image_pair(
        #         rendered_image[0], observed_image[0],
        #         "{}/rendered.png".format(settings.DEBUG_PATH),
        #         "{}/observed.png".format(settings.DEBUG_PATH),
        #         "{}/rendered_observed.png".format(settings.DEBUG_PATH))

        zoom_observed_image, zoom_observed_depth, zoom_observed_mask, zoom_rendered_image, zoom_rendered_depth, zoom_rendered_mask, interpolate_ratio = self.trim(
            observed_image, observed_depth, observed_mask, rendered_image,
            rendered_depth, rendered_mask, center_points, center_depth, **kwargs)

        data = torch.cat([
            zoom_observed_image, zoom_observed_depth, zoom_observed_mask,
            zoom_rendered_image, zoom_rendered_depth, zoom_rendered_mask
        ],
                         dim=1)
        return data, interpolate_ratio

    def state_encoding(self, data):
        state_feature, mask, flow = self.flownet(data)
        state_feature = F.adaptive_avg_pool2d(state_feature, 1).view(-1, 1024)

        return state_feature, mask, flow

    def action_encoding(self, state_feature, interpolate_ratio):
        mu, _ = self.action_encoding_mu_std(state_feature, interpolate_ratio)
        return mu

    def action_encoding_mu_std(self, state_feature, interpolate_ratio):
        mu, std = self.actor(state_feature, interpolate_ratio)
        return mu, std

    def reward_encoding(self, state_feature, action):
        return self.vdb(torch.cat([state_feature, action], dim=-1))

    def qlearning_encoding(self, state_feature, aciton):
        return self.vdb(torch.cat([state_feature, aciton], dim=-1))

    def forward(self, observed_image, observed_depth, observed_mask,
                init_pose: utils.Pose, mesh: Meshes, center_points,
                center_depth, **kwargs):
        data, interpolate_ratio = self.synthetic(observed_image,
                                                 observed_depth, observed_mask,
                                                 init_pose, mesh,
                                                 center_points, center_depth,
                                                 **kwargs)

        state_feature = self.state_encoding(data)

        return self.action_encoding(state_feature, interpolate_ratio)
