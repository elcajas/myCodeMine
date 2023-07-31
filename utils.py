import hashlib
import logging
from omegaconf import OmegaConf

import torch
import numpy as np
from itertools import product
import string

from mineclip import MineCLIP
from mineclip.mineagent.batch import Batch
import RewardEnv as Envs

import os
import sys
# sys.path.append("../../")

def set_MineCLIP(cfg, device):
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)

    assert (
        hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum
    ), "broken ckpt"

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt.path, strict=True)
    logging.info("MineCLIP successfully loaded with checkpoint")
    return model

def preprocess_obs(env_obs, device, prev_action, prompt, model):
    
    rgb_img = torch.tensor(env_obs['rgb'].copy()).float().to(device)
    with torch.no_grad():
        rgb_feat = model.clip_model.encode_image(rgb_img.unsqueeze(dim=0))
    # rgb_feat = torch.rand((1,512), device=device)
    obs = {
        "rgb_feat": rgb_feat,
        "compass": torch.tensor(np.append(
            np.sin(env_obs['location_stats']['pitch']*np.pi/180), [np.cos(env_obs['location_stats']['pitch']*np.pi/180),
            np.sin(env_obs['location_stats']['yaw']*np.pi/180), np.cos(env_obs['location_stats']['yaw']*np.pi/180)]), device=device).unsqueeze(dim=0),
        "gps": torch.tensor(env_obs['location_stats']['pos'], device=device).unsqueeze(dim=0),
        "voxels": torch.tensor(env_obs['voxels']['block_meta'], device=device).reshape(1,27), 
        "biome_id": torch.tensor(env_obs['location_stats']['biome_id'], device=device).unsqueeze(dim=0),
        "prev_action": torch.tensor(prev_action, device=device),
        "prompt": prompt.to(device),
    }
    frame = env_obs['rgb'].transpose(1,2,0)

    return Batch(obs=obs), frame

def complete_action(action, action_array):
    """
    Map agent action to env action.
    """
    assert action.ndim == 2
    action = action[0]
    action = action.cpu().numpy()

    return action, action_array[action.item()]

def actions_array(rango):  # Rango must be odd (3, 9)    3 -> (-15, 0 , +15)   9 -> (-60, -45, -30, -15, 0 , +15, +30, +45, +60)

    action_array = np.array([
        [1, 0, 0, 12, 12, 0, 0, 0],    # forward
        [1, 0, 1, 12, 12, 0, 0, 0],    # forward + jump
        [0, 0, 1, 12, 12, 0, 0, 0],    # jump
        [2, 0, 0, 12, 12, 0, 0, 0],    # backward
        [0, 1, 0, 12, 12, 0, 0, 0],    # left
        [0, 2, 0, 12, 12, 0, 0, 0],    # right
        [0, 0, 0, 12, 12, 3, 0, 0],    # use
        [0, 0, 0, 12, 12, 3, 0, 0],    # attack
    ])

    no_action = np.array([(rango*rango)//2 + 8])
    camera = np.zeros((rango*rango, 8))
    perm = product(np.arange(12 - rango//2, 12 + rango//2 + 1), repeat=2)
    camera[:,3:5] = np.array(list(perm))
    action_array = np.concatenate((action_array, camera), axis=0)

    return action_array, no_action

def create_env(task):
    step_penalty = 0
    nav_reward_scale = 1
    attack_reward = 5
    success_reward = 200
    env_dic = {
        'dummy env': 'MineEnv',
        'combat spider': 'CombatSpiderDenseRewardEnv',
        'combat zombie': 'CombatZombieDenseRewardEnv',
        'hunt a cow': 'HuntCowDenseRewardEnv',
        'hunt a sheep': 'HuntSheepDenseRewardEnv',
        'milk a cow': 'MilkCowDenseRewardEnv',
        'shear a sheep': 'ShearSheepDenseRewardEnv'
    }

    env_cls = getattr(Envs, env_dic[task])

    if 'combat' in task:
        env = env_cls(
            step_penalty=step_penalty,
            attack_reward=attack_reward,
            success_reward=success_reward
        )
        return env

    if 'milk' in task or 'shear' in task:    
        nav_reward_scale = 10
        attack_reward = 0

    env = env_cls(
        step_penalty=step_penalty,
        nav_reward_scale=nav_reward_scale,
        attack_reward=attack_reward,
        success_reward=success_reward,
    )
    return env

def KLD(P, Q):
    return (P * torch.log(P/Q)).sum(dim=1)

def smoothing(probs, W):
    i = W - 1
    tmp = 0
    for j in range(1, W):
        tmp += KLD(probs[i:], probs[(i-j):(-j)])
    return tmp.mean()

prompts = ["milk a cow", "shear a sheep", "hunt a cow", "hunt a sheep", "hunt a pig"]