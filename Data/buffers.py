import numpy as np
import logging
import torch
import pathlib
import imageio
from collections import namedtuple, deque

from mineclip.mineagent.batch import Batch
from .datasets import PPODataset

def discounted_cumsum(data, discount, tmp=0):
    sum = []
    for d in reversed(data):
        tmp = d + discount * tmp
        sum.append(tmp)
    return sum[::-1]

class PPOBuffer:
    def __init__(self, capacity, gamma, lam) -> None:
        self.states = [Batch(tmp=np.array([0]))] * capacity
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.state_values = np.zeros(capacity, dtype=np.float32)
        # self.is_terminals = np.zeros(capacity, dtype=np.float32)
        # self.next_states = [Batch(tmp=np.array([0]))] * capacity
        self.frames = []
        self.last_trajectory = Batch(tmp=np.array([0]))

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.pointer, self.path_start_idx, self.max_capacity = 0, 0, capacity 
    
    def store(self, state, action, reward, log_prob, state_value, frame):

        assert self.pointer < self.max_capacity

        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.log_probs[self.pointer] = log_prob
        self.state_values[self.pointer] = state_value
        # self.is_terminals[self.pointer] = is_terminal
        # self.next_states[self.pointer] = next_state
        self.frames.append(frame)
        self.pointer += 1

    def calc_advantages(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.pointer)
        rewards = self.rewards[path_slice]
        values = np.append(self.state_values[path_slice], last_val)
        trajectory = Batch.cat(self.states[path_slice])
        trajectory['actions'] = torch.tensor(self.actions[path_slice])
        trajectory['rewards'] = torch.tensor(self.rewards[path_slice])

        if trajectory.rewards.sum() >= 200:
            trajectory['successful'] = torch.tensor([1])
        else:
            trajectory['successful'] = torch.tensor([0])

        #GAE
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discounted_cumsum(deltas, self.gamma*self.lam)

        # Rewards-to-go
        self.returns[path_slice] = discounted_cumsum(rewards, self.gamma)

        self.path_start_idx = self.pointer
        self.last_trajectory = trajectory
        self.frames = []
        return trajectory

    def get(self, device):
        
        assert self.pointer == self.max_capacity

        self.pointer, self.path_start_idx = 0, 0

        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / adv_std
        data = dict(actions = self.actions,
                    rewards = self.rewards,
                    log_probs = self.log_probs,
                    state_values = self.state_values,
                    advantages = self.advantages,
                    returns = self.returns)
        
        data_dict = dict(states = Batch.cat(self.states))
        tmp_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        data_dict.update(tmp_dict)
        return PPODataset(data_dict, device)
    
    def make_video(self, path: pathlib.Path, reward):
        traj_slice = slice(self.path_start_idx, self.pointer)
        frames = self.frames
        
        logging.info(f"actions: {self.actions[traj_slice]}")
        logging.info(f"rewards: {self.rewards[traj_slice]}")
        logging.info(f"returns: {self.returns[traj_slice]}")

        video_dir = path.joinpath("videos")
        if not video_dir.exists():
            video_dir.mkdir()
        
        video_path = video_dir.joinpath(f"video_{reward:.2f}.mp4")
        writer = imageio.get_writer(video_path, fps=10)
        for frame in frames:
            writer.append_data(frame)
        writer.close()


class SIBuffer:
    def __init__(self, capacity) -> None:
        self.trajectories = [Batch(tmp=np.array([0]))] * capacity
        self.total_rewards = np.zeros(capacity, dtype=np.float32)
        self.probs = np.zeros(capacity, dtype=np.float32)

        self.mean, self.std = 0, 0
        self.traj_number, self.step_pos, self.max_capacity = 0, 0, capacity

    def store(self, trajectory):
        total_rew = trajectory.rewards.sum().item()
        threshold = self.mean + 2 * self.std

        if self.traj_number < self.max_capacity:
            if trajectory.successful == 1 or (total_rew > 0 and total_rew >= threshold):
                self.trajectories[self.traj_number] = trajectory
                self.total_rewards[self.traj_number] = total_rew
                self.traj_number += 1
                self.step_pos += len(trajectory.actions)

        else:
            pos = np.argmin(self.total_rewards)
            if trajectory.successful == 1 or (total_rew > 0 and total_rew >= threshold):
                self.trajectories[pos] = trajectory
                self.total_rewards[pos] = total_rew
    
    def sample(self):

        # Update mean and std
        self.mean = self.total_rewards[:self.traj_number].mean()
        self.std = np.std(self.total_rewards[:self.traj_number])
        thr = self.mean + 2 * self.std
        logging.info(f"mean: {self.mean:.2f}, std: {self.std:.2f}, threshold: {(thr):.2f}, num_traj: {self.traj_number}")

        return self.trajectories[:self.traj_number]


# Transition = namedtuple('Transition', 
#                         ('states', 'actions', 'rewards', 'log_probs', 'state_values', 'is_terminals', 'next_states'))
# Complete_transition = namedtuple('Complete_transition', 
#                         ('states', 'actions', 'rewards', 'log_probs', 'state_values', 'is_terminals', 'next_states', 'returns', 'advantages'))

# class MemoryReplay:
#     def __init__(self, capacity, gamma, lam) -> None:
#         self.memory = deque([], maxlen=capacity)
#         self.returns = deque([], maxlen=capacity)
#         self.advantages = deque([],maxlen=capacity)

#         self.gamma, self.lam = gamma, lam
#         self.pointer, self.path_start_idx = 0, 0
#         self.capacity = capacity

#     def add(self, *args):
#         self.memory.append(Transition(*args))
#         self.pointer += 1
    
#     def calc_advantages(self, last_val=0):
#         path_slice = slice(self.path_start_idx, self.pointer)
#         batch = Transition(*zip(*self.memory))
#         rewards = np.array((*batch.rewards[path_slice], last_val))
#         values = np.array((*batch.state_values[path_slice], last_val))

#     def __len__(self):
#         return len(self.memory)

# class RolloutBuffer:
#     def __init__(self) -> None:
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.state_values = []
#         self.is_terminals = []
    
#     def clear(self):
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.state_values[:]
#         del self.is_terminals[:]