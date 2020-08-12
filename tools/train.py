'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:16
'''
import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from glog import logger
from torch.utils.data import ConcatDataset
from tqdm import tqdm

import init_path
import trainer
import utils
from agent import Agent
from datasets import DataLoader, DataLoaderX, Synthetic, meshes_collate
from settings import SETTINGS
from visualize import visualize_losses

parser = argparse.ArgumentParser(description="Pose Agent Trainier")
parser.add_argument("--exname", default="PoseAgent", type=str)
parser.add_argument("--yaml_file", default="settings/ycb.yaml")
args = parser.parse_args()

settings = SETTINGS(yaml_file=args.yaml_file)
settings.merge_args(args)

# LOAD DATASET
train_dataset = []
if settings.DATASET == "ycb" or settings.DATASET == "all":
    from datasets import YCBDataset
    train_dataset.append(YCBDataset(settings, "train", settings.CLASS_ID))
train_dataloader = DataLoader(ConcatDataset(train_dataset),
                              batch_size=settings.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=meshes_collate,
                              num_workers=settings.NUM_WORKERS,
                              pin_memory=False,
                              drop_last=True)
if settings.IRL:
    # LOAD IRL DEMOSTRATION TODO
    pass
    demo_dataset = []
    demo_dataloader = []

agent = Agent(settings, device=settings.DEVICE).to(settings.DEVICE)
optimizer = optim.Adam([{
    "params": agent.flownet.parameters(),
    "lr": settings.BACKBONE_LEARNING_RATE
}, {
    "params": agent.actor.parameters()
}, {
    "params": agent.critic.parameters()
}, {
    "params": agent.vdb.parameters()
}],
                       lr=settings.LEARNING_RATE,
                       weight_decay=settings.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=settings.MILESTONES,
                                           gamma=settings.GAMMA)

# LOAD PRETRAINED
if os.path.exists(settings.PRETRAINED_MODEL):
    ckpt = torch.load(settings.PRETRAINED_MODEL, map_location=settings.DEVICE)
    utils.load_state_dict(agent.flownet, ckpt["state_dict"])
    logger.log(level=0,
               msg="Load pretrained model from {}".format(
                   settings.PRETRAINED_MODEL))
    print("Load pretrained model from {}".format(settings.PRETRAINED_MODEL))

# LOAD CHECKPOINT
if settings.RESUME:
    if os.path.isfile(settings.RESUME_PATH):
        ckpt = torch.load(settings.RESUME_PATH, map_location=settings.DEVICE)
        agent.load_state_dict(ckpt["model"])
        # optimizer.load_state_dict(ckpt["optimizer"])
        # scheduler.load_state_dict(ckpt["scheduler"])
        # start_epoch = ckpt["epoch"]
        start_epoch = settings.START_EPOCH
        # settings.set_episode(ckpt["episode"])
        # settings.set_synthetic_episode(ckpt["synthetic_episode"])
        logger.log(level=0, msg="Resume from {}.".format(settings.RESUME_PATH))
        print("Resume from {}.".format(settings.RESUME_PATH))
    else:
        raise FileExistsError("{} does not exist.".format(
            settings.RESUME_PATH))
else:
    start_epoch = settings.START_EPOCH
    logger.log(level=0, msg="Start a new train.")
    print("Start a new train.")

iteration = 0


def train(epoch):
    global iteration
    agent.train()
    # DEFINE DATALOADER HERE
    if settings.is_synthetic_dataset():
        syn_agent = Agent(settings, settings.SYNTHETIC_DEVICE).to(
            settings.SYNTHETIC_DEVICE)
        syn_agent.load_state_dict(agent.state_dict())
        syn_agent.eval()
        synthetic_dataloader = Synthetic(syn_agent,
                                         train_dataloader,
                                         settings=settings)
        synthetic_dataloader.start()
        loops = tqdm(synthetic_dataloader.fetch_data())
    else:
        loops = tqdm(train_dataloader)

    losses = []
    reward_losses = []
    for d in loops:
        if settings.is_synthetic_dataset():
            mesh = None
            raw_data = utils.variable_namedtuple(d, settings.DEVICE)
        else:
            mesh = d["mesh"].to(settings.DEVICE)
            raw_data = utils.variable_namedtuple(d["data"], settings.DEVICE)

        try:
            loss, reward_loss = trainer.__dict__[settings.TRAINER](agent, mesh,
                                                                   raw_data,
                                                                   optimizer,
                                                                   settings)
        except ZeroDivisionError as e:
            logger.log(level=1, msg=e)
            continue

        losses.append(loss)
        reward_losses.append(reward_loss)

        loops.set_description("Epoch: {}, Loss: {:.2f}".format(epoch, loss))
        iteration += 1
        if iteration % settings.LOG_INTERVAL == 0:
            visualize_losses({"loss": sum(losses) / len(losses)},
                             title="loss x {}".format(settings.LOG_INTERVAL),
                             env=settings.EXNAME,
                             iteration=iteration // settings.LOG_INTERVAL)
            losses = []
            visualize_losses(
                {"reward loss": sum(reward_losses) / len(reward_losses)},
                title="reward loss x {}".format(settings.LOG_INTERVAL),
                env=settings.EXNAME,
                iteration=iteration // settings.LOG_INTERVAL)
            reward_losses = []

            torch.save(
                {
                    "model": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "episode": settings.EPISODE_LEN,
                    "synthetic_episode": settings.SYNTHETIC_EPISODE_LEN,
                    "epoch": epoch
                },
                f=settings.SAVE_PATH)


def test(epoch):
    pass


if __name__ == "__main__":
    for epoch in range(start_epoch, settings.EPOCH):
        train(epoch)
        test(epoch)

        scheduler.step()
        settings.step_episode(
            inc=1
        )  # WARNING: If you are using multi-batch, the episode len while training should be 1. which means we do not rendering. If you want to do render, make sure the model in one batch is the same, or just keep the batch size === 1.
        settings.step_synthetic_episode(inc=1)

        # Save
        settings.RESUME = True
        settings.dump_yaml("settings/ycb.yaml")
