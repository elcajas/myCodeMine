import yaml
import os
import pickle
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
    if cfg_params.load_checkpoint:
        ppo_agent.load_model(cfg_params.checkpoint_dir, cfg_params.checkpoint_epoch)
    else:
        ppo_agent.save_model(res_path, 0)

    si_model = SImodel(cfg, device)
    mineCLIP = set_MineCLIP(cfg_mineclip, device)
    # ppo_agent.policy.load_state_dict(torch.load('/home/kenjic/documents/MineDojo_PPO/PPO_dummy/results/run_231012_1540/weights/model_epoch_10.pth'))

    task = cfg_params.task
    with torch.no_grad():
        prompt = mineCLIP.clip_model.encode_text(task)

    num_actions = cfg_params.number_actions
    rango = int((num_actions-8)**0.5)
    action_array, no_action = actions_array(rango)

    # no_action = np.array([4])

    env = create_env(task)
    state = env.reset()
    # ppo_agent.buffer.store_episode(state, 0, False, {})

    state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP)
    num_ep = 0
    steps_per_epoch = cfg_params.PPO_buffer_size
    epochs = cfg_params.epochs

    for epoch in range(epochs):              
        for t in range(steps_per_epoch):                                            # episode length

            action, log_prob, state_value = ppo_agent.process(state)
            action, env_action = complete_action(action, action_array)
            # action, env_action = select_action(t, action_array)
            # action, env_action = reduced_action(action)

            next_state, reward, done, _ = env.step(env_action)       #step(env_action)
            # ppo_agent.buffer.store_episode(next_state, reward, done, _)
            next_state, next_frame = preprocess_obs(next_state, device, action, prompt, mineCLIP)
            
            ppo_agent.buffer.store(state, action, reward, log_prob, state_value, frame)
            # ppo_agent.buffer.store_episode()

            # Update observation
            state, frame = next_state, next_frame

            timeout = (env._elapsed_steps == env._episode_len)
            terminal = (done or timeout)
            epoch_ended = (t == steps_per_epoch-1)

            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    logging.info(f'Warning: trajectory cut off by epoch at {env._elapsed_steps} steps.')
                
                if timeout or epoch_ended:
                    _, _, last_value = ppo_agent.process(state)
                    last_value = last_value.item()
                else:
                    last_value = 0
                
                ppo_agent.buffer.add_last_frame(frame)
                traj, trajectory = ppo_agent.buffer.calc_advantages(last_val=last_value, no_op=no_action.item())
                # si_model.buffer.store(trajectory)

                num_ep += 1
                ep_reward = trajectory.total_return.item()
                num_attack = (trajectory.actions == 7).sum().item()
                num_use = (trajectory.actions == 6).sum().item()

                logging.info(f'step: {t+1:6}, epoch: {epoch+1:2}, episode: {num_ep:3}, ep_rew: {ep_reward:.2f}, atk: {num_attack}, use: {num_use} ')
                wandb.log({"episode_reward": ep_reward, "attack": num_attack, "use": num_use, "episode": num_ep,})

                # ppo_agent.buffer.make_video(res_path, 0.0, 'none')
                if ep_reward >= cfg_params.video_min_rew:
                    logging.info(f'making video with reward {ep_reward:.2f}')
                    ppo_agent.buffer.make_video(res_path, ep_reward, num_ep)
                    ppo_agent.buffer.get_images(res_path, ep_reward, num_ep)
                
                # if ep_reward >= 200:
                #     logging.info(f'saving successful episode')
                #     save_object(traj, 'episode_dict.pkl')
                #     return

                state = env.reset()
                # ppo_agent.buffer.store_episode(state, 0, False, {})
                state, frame = preprocess_obs(state, device, no_action, prompt, mineCLIP)

        # Update PPO after buffer is full
        ppo_agent.update()

        # Self Imitation learning training
        # if si_model.buffer.traj_number > 0:
        #     weights = ppo_agent.policy.actor.state_dict()
        #     si_model.train(weights=weights)
        #     ppo_agent.policy.actor.load_state_dict(si_model.model.state_dict())

        ppo_agent.save_model(res_path, epoch)
    env.close()

def init_log(cfg, res_path):
    filename = res_path.joinpath(f"output.log") if cfg.file_logging else None
    logging.basicConfig(
        filename=filename,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode= 'w'
    )
    
    if not cfg.wandb_init: os.environ["WANDB_MODE"] = "disabled"
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"MineDojo-PPO-SI_{cfg.task.replace(' ','')}",
        # group=f"action{cfg.number_actions}",
        name=f"actions_{cfg.number_actions}_mean{cfg.buffer_mean}_std{cfg.buffer_std}_delta{cfg.buffer_delta}_batch{cfg.batch_size}_IL{cfg.imitation_learning}",
        
        # track hyperparameters and run metadata
        config= dict(cfg)
    )

def save_object(obj, filename):
    dirpth = dir_path.joinpath(filename)
    with open(dirpth, 'wb') as outp:                      # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()