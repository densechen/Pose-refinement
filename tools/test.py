'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:14
'''
import argparse
import json
import os

import numpy as np
import scipy.io as scio
import torch
from glog import logger
from torch.utils.data import ConcatDataset
from tqdm import tqdm

import init_path
import utils
from agent import Agent
from datasets import DataLoaderX, meshes_collate
from settings import SETTINGS
from visualize import visualize_scatters

parser = argparse.ArgumentParser(description="Pose Agent Tester")
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--device", default="0", type=str)
parser.add_argument("--exname", default="PaoseAgent", type=str)
parser.add_argument("--yaml_file", default="settings/ycb.yaml")
args = parser.parse_args()

settings = SETTINGS(yaml_file=args.yaml_file)
settings.merge_args(args)

# LOAD DATASET
if settings.DATASET == "ycb" or settings.DATASET == "all":
    from datasets import YCBTestDataset
    test_dataset = YCBTestDataset(settings, "test", settings.CLASS_ID)

agent = Agent(settings, device=settings.DEVICE).to(settings.DEVICE)
# LOAD CHECKPOINTS
if os.path.isfile(settings.RESUME_PATH):
    ckpt = torch.load(settings.RESUME_PATH, map_location=settings.DEVICE)
    agent.load_state_dict(ckpt["model"])
    logger.log(level=0, msg="Resume from {}.".format(settings.RESUME_PATH))
    print("Load model from {}.".format(settings.RESUME_PATH))
else:
    raise FileExistsError("{} does not exist.".format(settings.RESUME_PATH))


def test():
    loops = tqdm(range(len(test_dataset)))

    for index in loops:
        predict_pose = []
        for d in test_dataset.load_data(index):
            # TODO: CONVDERT THIS TEST FILE TO BATCH TEST MODE
            try:
                d = meshes_collate([d])
                mesh = d["mesh"].to(settings.DEVICE)
                raw_data = utils.variable_namedtuple(d["data"],
                                                     settings.DEVICE)
                source_pose = raw_data.init_pose
                intrinsic = raw_data.Intrinsic

                settings.set_intrinsic(intrinsic)
                with torch.no_grad():
                    for te in range(settings.TEST_EPISODE):
                        center_points, center_depth = utils.translation_to_voxel_and_depth(
                            source_pose.Translation.translation, intrinsic,
                            settings)
                        data, interpolate_ratio = agent.synthetic(
                            observed_image=raw_data.image,
                            observed_depth=raw_data.depth,
                            observed_mask=raw_data.mask,
                            init_pose=source_pose,
                            mesh=mesh,
                            center_points=center_points,
                            center_depth=center_depth,
                            settings=settings,
                            time=te)
                        state_feature, _, _ = agent.state_encoding(data)
                        action = agent.action_encoding(state_feature,
                                                       interpolate_ratio)
                        if settings.WITH_REWARD:
                            reward = agent.reward_encoding(
                                state_feature, action)
                            if reward < 0.02:  # This is a bad pose, stop here.
                                break

                        source_pose = utils.apply_action_to_pose(
                            action, source_pose, settings)
                # premute
                rotation = source_pose.Rotation.matrix.permute(0, 2, 1)
                rotation = utils.build_rotation(rotation, format="matrix")
                p = torch.cat(
                    [rotation.quat, source_pose.Translation.translation],
                    dim=-1).view(7).cpu().numpy().tolist()
                predict_pose.append(p)
            except Exception as e:
                print(e)
                predict_pose.append([0.0 for i in range(7)])

        scio.savemat(
            "{}/{}.mat".format(settings.RESULT_DIR, "{:04d}".format(index)),
            {"poses": predict_pose})
        loops.set_description("Finish No. {} keyframe".format(index))


if __name__ == "__main__":
    with torch.no_grad():
        agent.eval()
        test()
