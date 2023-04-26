import hydra
import hashlib
from omegaconf import OmegaConf

import numpy as np
import torch
from tqdm import tqdm
from itertools import product

import imageio
import pickle
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
from models import PPO

import pdb

# dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = pathlib.Path(__file__).parent.resolve()
res_path = dir_path.joinpath("results")
time = datetime.datetime.now().strftime("%y%m%d_%H%M")
if not res_path.exists():
    res_path.mkdir()
log_path = res_path.joinpath(f"results_{time}.log")

logging.basicConfig(filename=log_path,
                    format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
                    filemode= 'w')

wandb.init(
    # set the wandb project where this run will be logged
    project="MineDojo-PPO",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "PPO",
    "environment": "HuntCowDenseReward",
    "epochs": 20,
    }
)

# os.environ['IMAGEIO_FFMPEG_EXE'] = '/opt/homebrew/bin/ffmpeg'
## DUMMY ENV
class MineEnv:
    def __init__(self):
        self.val = 0
        self._elapsed_steps = 0
        self._episode_len = 500
        with open(dir_path.joinpath("observation_dict.pkl"), 'rb') as f:
            self.obs = pickle.load(f)

    def reset(self):
        self._elapsed_steps = 0
        return self.obs

    def step(self, action):
        self._elapsed_steps += 1
        return self.obs, 1, False, 1

    def close(self):
        return
    
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

    obs = {
        "rgb_feat": rgb_feat,
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
        [1, 0, 0, 12, 12, 0, 0, 0],
        [1, 0, 1, 12, 12, 0, 0, 0],
        [0, 0, 1, 12, 12, 0, 0, 0],
        [2, 0, 0, 12, 12, 0, 0, 0],
        [0, 1, 0, 12, 12, 0, 0, 0],
        [0, 2, 0, 12, 12, 0, 0, 0],
        [0, 0, 0, 12, 12, 1, 0, 0],
        [0, 0, 0, 12, 12, 3, 0, 0],
    ])
no_action = np.array([48])
camera = np.zeros((81, 8))
perm = product(np.arange(8,17), repeat=2)
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

@hydra.main(config_name="config", config_path=".", version_base="1.1")
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    OmegaConf.set_struct(cfg, False)
    cfg_mineclip = cfg.pop("mineclip")
    OmegaConf.set_struct(cfg, True)

    ppo_agent = PPO(cfg, device)
    mineCLIP = set_MineCLIP(cfg_mineclip, device)

    prompt = "Hunt a cow"
    with torch.no_grad():
        prompt = mineCLIP.clip_model.encode_text(prompt)
    # env = MineEnv()
    env = HuntCowDenseRewardEnv(
        step_penalty=0,
        attack_reward=1,
        nav_reward_scale=5,
        success_reward=200,
    )

    state = env.reset()
    state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP)
    
    ep_reward, total_timesteps, num_ep = 0, 0, 0
    steps_per_epoch = 100000
    epochs = 20

    for epoch in range(epochs):              
        for t in range(steps_per_epoch):                 # episode length
            num_attack = 0
            action, log_prob, state_value = ppo_agent.process(state)
            if action.item() == 7: num_attack += 1
            action, env_action = complete_action(action)

            next_state, reward, done, _ = env.step(env_action)

            next_state, next_frame = preprocess_obs(next_state, device, action, prompt, mineCLIP)
            total_timesteps += 1
            ep_reward += reward

            ppo_agent.buffer.store(state, action, reward, log_prob, state_value, done, next_state, frame)
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

                if ep_reward > 200:
                    ppo_agent.buffer.make_video(dir_path, ep_reward)
                    print('making video')
                ppo_agent.buffer.calc_advantages(last_val=last_value)
                num_ep += 1
                logging.info(f'step: {t+1} epoch: {epoch+1} episode: {num_ep}  episode reward: {ep_reward} attacked:{num_attack} times')
                wandb.log({"episode_reward": ep_reward, "attack_action": num_attack, "episode": num_ep})
                state = env.reset()
                state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP)
                ep_reward, ep_len = 0, 0

        #Update PPO after buffer is full
        ppo_agent.update()
        ppo_agent.save_model(res_path, time, epoch)
    env.close()

if __name__ == "__main__":
    main()