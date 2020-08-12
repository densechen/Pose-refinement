'''
Descripttion: densechen@foxmail.com
version: 0.0
Author: Dense Chen
Date: 1970-01-01 08:00:00
LastEditors: Dense Chen
LastEditTime: 2020-08-12 20:45:34
'''
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

import utils


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy


def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)


def get_reward(discrim, state, action):
    with torch.no_grad():
        return -torch.log(discrim(torch.cat([state, action], dim=-1)))


def train_discrim(discrim, state_features, actions, optim, demostrations,
                  settings):
    """demostractions: [state_features|actions]
    """
    criterion = torch.nn.BCELoss()

    for _ in range(settings.VDB_UPDATE_NUM):
        learner = discrim(torch.cat([state_features, actions], dim=-1))
        expert = discrim(demostrations)

        discrim_loss = criterion(learner, torch.ones(
            [len(state_features), 1])) + criterion(
                expert, torch.zeros(len(demostrations), 1))
        optim.zero_grad()
        discrim_loss.backward()
        optim.step()

    expert_acc = ((discrim(demostrations) < 0.5).float()).mean()
    learner_acc = ((discrim(torch.cat([state_features, actions], dim=1)) >
                    0.5).float()).mean()

    return expert_acc, learner_acc


def get_gae(rewards, masks, values, settings):
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (settings.IRL_GAMMA * running_returns *
                                        masks[t])
        returns = running_returns

        running_delta = rewards[t] + (settings.IRL_GAMMA * previous_value *
                                      masks[t]) - values.data[t]
        previous_value = values.data[t]

        running_advants = running_delta + (settings.IRL_GAMMA *
                                           settings.IRL_LAMDA *
                                           running_advants * masks[t])
        advants[t] = running_advants
    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, interpolate_ratio, old_policy,
                   actions, batch_index):
    mu, std = actor(states, interpolate_ratio)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy


def train_actor_critic(actor, critic, state_features, interpolate_ratio,
                       actions, rewards, masks, optim, settings):
    old_values = critic(state_features)
    returns, advants = get_gae(rewards, masks, old_values, settings)

    mu, std = actor(state_features, interpolate_ratio)
    old_policy = log_prob_density(actions, mu, std)

    criterion = torch.nn.MSELoss()
    n = len(state_features)
    arr = np.arange(n)

    losses = []

    for _ in range(settings.ACTOR_CRITIC_UPDATE_NUM):
        np.random.shuffle(arr)

        for i in range(n // settings.BATCH_SIZE):
            batch_index = arr[settings.BATCH_SIZE * i:settings.BATCH_SIZE *
                              (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = state_features[batch_index]
            inputs_interpolate_ratio = interpolate_ratio[batch_index]
            actions_samples = actions[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalues_samples = old_values[batch_index].detach()

            values = critic(inputs)
            clipped_values = oldvalues_samples + torch.clamp(
                values - oldvalues_samples, -settings.CLIP_PARAM,
                settings.CLIP_PARAM)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples,
                                                  inputs,
                                                  inputs_interpolate_ratio,
                                                  old_policy.detach(),
                                                  actions_samples, batch_index)
            clipped_ratio = torch.clamp(ratio, 1.0 - settings.CLIP_PARAM,
                                        1.0 + settings.CLIP_PARAM)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()
            losses.append(loss.item())

        return sum(losses) / len(losses)