import hashlib
import logging
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import numpy as np
from itertools import product
import string

from mineclip import MineCLIP
from mineclip.mineagent.batch import Batch
import RewardEnv as Envs

import groundingdino.datasets.transforms as T
from PIL import Image
from inference import load_model, load_image, predict, annotate
import cv2

import os
import sys
# sys.path.append("../../")

# calculated from 21K video clips, which contains 2.8M frames
MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

def set_gdino(cfg, device):
    model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
    return model

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

def preprocess_image(images, model, device):
    if isinstance(model, MineCLIP):
        img_tensor = torch.from_numpy(images).to(device)
        with torch.no_grad():
            return model.forward_image_features(img_tensor.unsqueeze(dim=0))
    else:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(MC_IMAGE_MEAN, MC_IMAGE_STD),
            ]
        )
        img_array = images.transpose(1,2,0)
        img = Image.fromarray(img_array)
        img_transformed, _ = transform(img, None)
        
        TEXT_PROMPT = "spider . cow . sky . animal . tree ."
        logits = predict(
            model=model,
            image=img_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        # annotated_frame = annotate(image_source=img_array, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite("annotated_image.jpg", annotated_frame)
        return logits


def preprocess_obs(env_obs, device, prev_action, prompt, model):

    image = env_obs['rgb'].copy()
    frame = image.transpose(1,2,0)
    rgb_feat = preprocess_image(image, model, device)

    obs = {
        "rgb_feat": rgb_feat,
        "compass": torch.tensor(np.append(
            np.sin(env_obs['location_stats']['pitch']*np.pi/180), [np.cos(env_obs['location_stats']['pitch']*np.pi/180),
            np.sin(env_obs['location_stats']['yaw']*np.pi/180), np.cos(env_obs['location_stats']['yaw']*np.pi/180)]), device=device).unsqueeze(dim=0),
        "gps": torch.tensor(env_obs['location_stats']['pos'], device=device).unsqueeze(dim=0),
        "voxels": torch.tensor(env_obs['voxels']['block_meta'], device=device).reshape(1,27), 
        "biome_id": torch.tensor(env_obs['location_stats']['biome_id'], device=device).unsqueeze(dim=0),
        "prev_action": torch.tensor(prev_action, device=device),
        "prompt": prompt,
    }

    return Batch(obs=obs), frame

def complete_action(action, action_array):
    """
    Map agent action to env action.
    """
    assert action.ndim == 2
    action = action[0]
    action = action.cpu().numpy()

    return action, action_array[action.item()]

def select_action(t, action_array):
    t = t % 13
    if t > 0:
        action = np.array([49])
    else:
        action = np.array([7])

    return action, action_array[action.item()]

def reduced_action(action):
    action_array = np.array([
        [0, 0, 0, 12, 12, 1, 0, 0],    # use
        [0, 0, 0, 12, 12, 3, 0, 0],    # attack
        [0, 0, 0, 12, 13, 0, 0, 0],    # rotate right
        [0, 0, 0, 12, 11, 0, 0, 0],    # rotate left
        [0, 0, 0, 12, 12, 0, 0, 0],    # no_op
        [0, 0, 0, 12, 12, 0, 0, 0],    # no_op
        [0, 0, 0, 12, 12, 0, 0, 0],    # no_op
        [0, 0, 0, 12, 12, 0, 0, 0],    # no_op
    ])
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
        [0, 0, 0, 12, 12, 1, 0, 0],    # use
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