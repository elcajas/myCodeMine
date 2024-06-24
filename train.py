import hydra
import hashlib
import yaml
from omegaconf import OmegaConf

import numpy as np
import torch
import torchvision
import cv2
from tqdm import tqdm
from itertools import product

import pathlib
import logging
import wandb
import datetime
import os
import sys
# sys.path.append("../../")
from mineclip import MineCLIP
from mineclip.mineagent.batch import Batch
from mineclip import CombatSpiderDenseRewardEnv
from mineclip import HuntCowDenseRewardEnv

from RewardEnv.milk_cow import MilkCowDenseRewardEnv
from RewardEnv.dummy_env import MineEnv
from models import PPO, SImodel

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import pdb

time = datetime.datetime.now().strftime("%y%m%d_%H%M")
dir_path = pathlib.Path(__file__).parent.resolve()
res_path = dir_path.joinpath(f"results/run_{time}")
res_path.mkdir(parents=True, exist_ok=True)

log_path = res_path.joinpath(f"output.log")
logging.basicConfig(
    filename=log_path,
    format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    filemode= 'w',
)
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="MineDojo-PPO-SI-init",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 1e-4,
#     "architecture": "PPO_SI",
#     "environment": "HuntCowDenseReward",
#     "epochs": 10,
#     "initial_mean": 20,
#     "num_actions": 89,
#     }
# )

def segment_image(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = image[m].mean(axis=0)/255
        img[m] = color_mask
    return img

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

def set_SamModel():
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.89,
        stability_score_thresh=0.96,
    )   
    return mask_generator

def preprocess_obs(env_obs, device, prev_action, prompt, model, model2):
    
    rgb_img = torch.tensor(env_obs['rgb'].copy()).float().to(device)

    # image = env_obs['rgb'].copy().transpose(1,2,0)
    # model2.predictor.set_image(image)
    # mask_feat = model2.predictor.get_image_embedding().flatten(start_dim=1)

    # anns_img = model2.generate(image)
    # mask_img = segment_image(anns_img, image).transpose(2,0,1)
    # mask_img = torch.tensor(mask_img).float().to(device)

    with torch.no_grad():
        rgb_feat = model.clip_model.encode_image(rgb_img.unsqueeze(dim=0))
        # mask_feat = model.clip_model.encode_image(mask_img.unsqueeze(dim=0))
    # rgb_feat = torch.rand((1,512), device=device)
    obs = {
        "rgb_feat": rgb_feat,
        # "mask_feat": mask_feat,
        "compass": torch.tensor(np.append(
            np.sin(env_obs['location_stats']['pitch']), [np.cos(env_obs['location_stats']['pitch']),
            np.sin(env_obs['location_stats']['yaw']), np.cos(env_obs['location_stats']['yaw'])]), device=device).unsqueeze(dim=0),
        "gps": torch.tensor(env_obs['location_stats']['pos'], device=device).unsqueeze(dim=0),
        "voxels": torch.tensor(env_obs['voxels']['block_meta'], device=device).reshape(1,27), 
        "biome_id": torch.tensor(env_obs['location_stats']['biome_id'], device=device).unsqueeze(dim=0),
        "prev_action": torch.tensor(prev_action, device=device),
        "prompt": prompt.to(device),
    }
    frame = env_obs['rgb'].transpose(1,2,0)

    return Batch(obs=obs), frame

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
rango = 9 # must be odd number
no_action = np.array([(rango*rango)//2 + 8])
camera = np.zeros((rango*rango, 8))
perm = product(np.arange(12 - rango//2, 12 + rango//2 + 1), repeat=2)
camera[:,3:5] = np.array(list(perm))
action_array = np.concatenate((action_array, camera), axis=0)

def complete_action(action):
    """
    Map agent action to env action.
    """
    assert action.ndim == 2
    action = action[0]
    action = action.cpu().numpy()

    return action, action_array[action.item()]

# @hydra.main(config_name="config", config_path=".", version_base="1.1")
def main():

    with open(dir_path.joinpath("config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    
    OmegaConf.set_struct(cfg, False)
    cfg_mineclip = cfg.pop("mineclip")
    OmegaConf.set_struct(cfg, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_agent = PPO(cfg, device)
    si_model = SImodel(cfg, device)
    mineCLIP = set_MineCLIP(cfg_mineclip, device)
    sam = set_SamModel()

    prompt = "milk a cow"               # possible prompts: milk a cow, shear a sheep, hunt a cow, hunt a sheep, combat {monster}
    with torch.no_grad():
        prompt = mineCLIP.clip_model.encode_text(prompt)
    # env = MineEnv()
    # env = HuntCowDenseRewardEnv(
    #     step_penalty=0,
    #     nav_reward_scale=1,
    #     attack_reward=5,
    #     success_reward=200,
    # )
    env = MilkCowDenseRewardEnv(
        step_penalty=0,
        nav_reward_scale=10,
        attack_reward=0,
        success_reward=200,
    )

    state = env.reset()
    state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP, sam)
    
    ep_reward, total_timesteps, num_ep, num_attack, num_use = 0, 0, 0, 0, 0
    steps_per_epoch = 500*200
    epochs = 1

    ppo_agent.policy.load_state_dict(torch.load('/home/user/Mine/results/run_230607_0158/weights/model_epoch_6.pth'))
    # weights = ppo_agent.policy.actor.state_dict()
    # si_model.train(weights=weights)
    # ppo_agent.policy.actor.load_state_dict(si_model.model.state_dict())

    for epoch in range(epochs):              
        for t in range(steps_per_epoch):                 # episode length
            action, log_prob, state_value = ppo_agent.process(state)
            if action.item() == 7:
                num_attack += 1
            if action.item() == 6:
                num_use += 1
            action, env_action = complete_action(action)

            next_state, reward, done, _ = env.step(env_action)

            next_state, next_frame = preprocess_obs(next_state, device, action, prompt, mineCLIP, sam)
            total_timesteps += 1
            ep_reward += reward

            ppo_agent.buffer.store(state, action, reward, log_prob, state_value, frame)
            # ppo_agent.buffer.rewards.append(reward)
            # ppo_agent.buffer.is_terminals.append(done)

            # Update observation
            state, frame = next_state, next_frame
            timeout = (env._elapsed_steps == env._episode_len)
            terminal = (done or timeout)
            epoch_ended = (t == steps_per_epoch-1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    logging.info(f'Warning: trajectory cut off by epoch at {ep_len} steps.', flush=True)

                if timeout or epoch_ended:
                    _, _, last_value = ppo_agent.process(state)
                    last_value = last_value.item()
                else:
                    last_value = 0
                    
                num_ep += 1
                trajectory = ppo_agent.buffer.calc_advantages(last_val=last_value)

                if trajectory.total_return > 10:
                    logging.info('making video')
                    ppo_agent.buffer.make_video(res_path, ep_reward, num_ep)

                si_model.buffer.store(trajectory)
                logging.info(f'step: {t+1}, epoch: {epoch+1}, episode: {num_ep}, ep_rew: {ep_reward:.2f}, atk: {num_attack}, use: {num_use} ')
                # wandb.log({
                #     "episode_reward": ep_reward,
                #     "attack": num_attack,
                #     "use": num_use,
                #     "episode": num_ep,
                # })
                state = env.reset()
                state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP, sam)
                ep_reward, ep_len, num_attack, num_use = 0, 0, 0, 0

        # Update PPO after buffer is full
        # ppo_agent.update()

        # # Self Imitation learning training
        # if si_model.buffer.traj_number > 0:
        #     weights = ppo_agent.policy.actor.state_dict()
        #     si_model.train(weights=weights)
        #     ppo_agent.policy.actor.load_state_dict(si_model.model.state_dict())

        # ppo_agent.save_model(res_path, epoch)

    env.close()

if __name__ == "__main__":
    main()