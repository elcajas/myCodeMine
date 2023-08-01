import yaml
import os
from omegaconf import OmegaConf

import pathlib
import logging
import wandb
import datetime

import torch
from models import PPO, SImodel
from utils import *

time = datetime.datetime.now().strftime("%y%m%d_%H%M")
dir_path = pathlib.Path(__file__).parent.resolve()
res_path = dir_path.joinpath(f"results/run_{time}")
res_path.mkdir(parents=True, exist_ok=True)

def main():
    with open(dir_path.joinpath("config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg_mineclip = cfg.pop("mineclip")
    OmegaConf.set_struct(cfg, True)
    cfg_params = cfg.hyperparameters
    init_log(cfg_params, res_path)                                                  #Initialize logging and wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_agent = PPO(cfg, device)
    si_model = SImodel(cfg, device)
    mineCLIP = set_MineCLIP(cfg_mineclip, device)

    task = cfg_params.task
    with torch.no_grad():
        prompt = mineCLIP.clip_model.encode_text(task)

    action_array, no_action = actions_array(rango=9)
    env = create_env(task)
    state = env.reset()
    state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP)
    
    num_ep = 0
    steps_per_epoch = cfg_params.PPO_buffer_size
    epochs = cfg_params.epochs

    for epoch in range(epochs):              
        for t in range(steps_per_epoch):                                            # episode length

            action, log_prob, state_value = ppo_agent.process(state)
            action, env_action = complete_action(action, action_array)

            next_state, reward, done, _ = env.step(env_action)
            next_state, next_frame = preprocess_obs(next_state, device, action, prompt, mineCLIP)

            ppo_agent.buffer.store(state, action, reward, log_prob, state_value, frame)

            # Update observation
            state, frame = next_state, next_frame

            timeout = (env._elapsed_steps == env._episode_len)
            terminal = (done or timeout)
            epoch_ended = (t == steps_per_epoch-1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    logging.info(f'Warning: trajectory cut off by epoch at {t} steps.')

                if timeout or epoch_ended:
                    _, _, last_value = ppo_agent.process(state)
                    last_value = last_value.item()
                else:
                    last_value = 0
                    
                trajectory = ppo_agent.buffer.calc_advantages(last_val=last_value)
                si_model.buffer.store(trajectory)

                num_ep += 1
                ep_reward = trajectory.total_return.item()
                num_attack = (trajectory.actions == 7).sum().item()
                num_use = (trajectory.actions == 6).sum().item()

                if ep_reward > cfg_params.video_min_rew:
                    logging.info('making video')
                    ppo_agent.buffer.make_video(res_path, ep_reward, num_ep)

                logging.info(f'step: {t+1}, epoch: {epoch+1}, episode: {num_ep}, ep_rew: {ep_reward:.2f}, atk: {num_attack}, use: {num_use} ')
                wandb.log({
                    "episode_reward": ep_reward,
                    "attack": num_attack,
                    "use": num_use,
                    "episode": num_ep,
                })
                state = env.reset()
                state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP)

        # Update PPO after buffer is full
        ppo_agent.update()

        # Self Imitation learning training
        if si_model.buffer.traj_number > 0:
            weights = ppo_agent.policy.actor.state_dict()
            si_model.train(weights=weights)
            ppo_agent.policy.actor.load_state_dict(si_model.model.state_dict())

        ppo_agent.save_model(res_path, epoch)
    env.close()

def init_log(cfg, res_path):
    filename = res_path.joinpath(f"output.log")
    if not cfg.file_logging:
        filename = None
    logging.basicConfig(filename=filename,
                    format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    filemode= 'w')
    
    if not cfg.wandb_init:
        os.environ["WANDB_MODE"] = "disabled"

    wandb.init(
        # set the wandb project where this run will be logged
        project=f"MineDojo-PPO-SI_{cfg.task.replace(' ','')}",
        group=f"action{cfg.number_actions}",
        name=f"mean{cfg.buffer_mean}_std{cfg.buffer_std}_delta{cfg.buffer_delta}_batch{cfg.batch_size}_IL{cfg.imitation_learning}",
        
        # track hyperparameters and run metadata
        config= dict(cfg)
    )

if __name__ == "__main__":
    main()