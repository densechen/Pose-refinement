'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:44:27
'''
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, Queue, set_start_method
from torch.utils.data import ConcatDataset

import utils
from datasets import DataLoader, DataLoaderX, meshes_collate

try:
    set_start_method('spawn')
except RuntimeError:
    pass


class Synthetic(Process):
    def __init__(self, agent, dataloader, settings):
        super().__init__()

        self.agent = agent
        self.dataloader = dataloader
        self.settings = settings

        self.queue = Queue(maxsize=settings.QUEUE_LEN)
        self.put_flag = Queue(maxsize=1)
        self.get_flag = Queue(maxsize=1)
        self.done = False

    def update_settings(self, settings):
        self.settings = settings

    def update_agent(self, target_agent):
        self.agent.load_state_dict(target_agent.state_dict())

    def fetch_data(self):
        num_batch = self.settings.NUM_BATCH_WHILE_SYNTHETIC
        while self.put_flag.empty():
            out = []
            for _ in range(num_batch):
                d = self.queue.get()
                if self.queue.qsize() < num_batch:
                    self.queue.put(d)
                out.append(d)
            yield utils.cat_namedtuple_list(out, dim=0)

        # Put a single to flag
        self.get_flag.put(True)

    def run(self):
        """ Generate Data Queue
        """
        settings = self.settings
        for d in self.dataloader:
            episode_data, episode_interpolate_ratio, episode_source_pose = [], [], []

            mesh = d["mesh"].to(settings.SYNTHETIC_DEVICE)
            raw_data = utils.variable_namedtuple(d["data"],
                                                 settings.SYNTHETIC_DEVICE)

            source_pose = raw_data.init_pose
            target_pose = raw_data.target_pose
            intrinsic = raw_data.Intrinsic
            settings.set_intrinsic(intrinsic)

            for _ in range(settings.SYNTHETIC_EPISODE_LEN):
                episode_source_pose.append(source_pose)
                center_points, center_depth = utils.translation_to_voxel_and_depth(
                    source_pose.Translation.translation, intrinsic,
                    self.settings)
                try:
                    syn_data, interpolate_ratio = self.agent.synthetic(
                        observed_image=raw_data.image,
                        observed_depth=raw_data.depth,
                        observed_mask=raw_data.mask,
                        init_pose=source_pose,
                        mesh=mesh,
                        center_points=center_points,
                        center_depth=center_depth,
                        settings=settings)
                    if settings.SYNTHETIC_EPISODE_LEN > 1:
                        state_feature, mask, flow = self.agent.state_encoding(
                            syn_data)
                        action = self.agent.action_encoding(
                            state_feature, interpolate_ratio)
                        source_pose = utils.apply_action_to_pose(
                            action, source_pose, settings)
                        source_pose = utils.detach_namedtuple(source_pose)
                    episode_data.append(syn_data)
                    episode_interpolate_ratio.append(interpolate_ratio)
                except Exception as e:
                    print(e)
            if len(episode_data) != settings.SYNTHETIC_EPISODE_LEN or len(
                    episode_interpolate_ratio
            ) != settings.SYNTHETIC_EPISODE_LEN:
                # Something may be wrong while generating data
                continue
            # append data to queue
            for i in range(settings.SYNTHETIC_EPISODE_LEN):
                syn_raw_data = utils.SynRawData(
                    data=episode_data[i],
                    Intrinsic=intrinsic,
                    target_pose=target_pose,
                    init_pose=episode_source_pose[i],
                    model_points=raw_data.model_points,
                    interpolate_ratio=episode_interpolate_ratio[i])
                syn_raw_data = utils.variable_namedtuple(syn_raw_data,
                                                         device="cpu")
                self.queue.put(syn_raw_data)
        # Put a single to flag
        self.put_flag.put(True)
        # Waiting for main thread finish last data fetch
        while self.get_flag.empty():
            time.sleep(2)
