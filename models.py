import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import imageio
import pathlib
import wandb

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from mineclip.mineagent import features as F
from mineclip import SimpleFeatureFusion, MineAgent, MultiCategoricalActor
from mineclip.mineagent.batch import Batch

from mineclip.utils import build_mlp
import pdb

def discounted_cumsum(data, discount, tmp=0):
    sum = []
    for d in reversed(data):
        tmp = d + discount * tmp
        sum.append(tmp)
    return sum[::-1]
        
Transition = namedtuple('Transition', 
                        ('states', 'actions', 'rewards', 'log_probs', 'state_values', 'is_terminals', 'next_states'))
Complete_transition = namedtuple('Complete_transition', 
                        ('states', 'actions', 'rewards', 'log_probs', 'state_values', 'is_terminals', 'next_states', 'returns', 'advantages'))

class MemoryReplay:
    def __init__(self, capacity, gamma, lam) -> None:
        self.memory = deque([], maxlen=capacity)
        self.returns = deque([], maxlen=capacity)
        self.advantages = deque([],maxlen=capacity)

        self.gamma, self.lam = gamma, lam
        self.pointer, self.path_start_idx = 0, 0
        self.capacity = capacity

    def add(self, *args):
        self.memory.append(Transition(*args))
        self.pointer += 1
    
    def calc_advantages(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.pointer)
        batch = Transition(*zip(*self.memory))
        rewards = np.array((*batch.rewards[path_slice], last_val))
        values = np.array((*batch.state_values[path_slice], last_val))

    def __len__(self):
        return len(self.memory)

class RolloutBuffer:
    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPODataset(Dataset):
    def __init__(self, data: dict, device: torch.device) -> None:
        self.state_data = data['states']
        self.action_data = data['actions'].to(device)
        self.advantage_data = data['advantages'].to(device)
        self.old_logp_data = data['log_probs'].to(device)
        self.return_data = data['returns'].to(device)
    
    def __len__(self):
        return len(self.action_data)
    
    def __getitem__(self, index):
        
        state = self.state_data[index]
        action = self.action_data[index]
        advantage = self.advantage_data[index]
        old_logp = self.old_logp_data[index]
        retur = self.return_data[index]

        return state, action, advantage, old_logp, retur

def custom_collate_fn(batch):
    arranged_batch = list(zip(*batch))
    state_batch = Batch.stack(arranged_batch[0])
    action_batch = torch.stack(arranged_batch[1])
    advantage_batch = torch.stack(arranged_batch[2])
    old_logp_batch = torch.stack(arranged_batch[3])
    return_batch = torch.stack(arranged_batch[4])

    return state_batch, action_batch, advantage_batch, old_logp_batch, return_batch

class PPOBuffer:
    def __init__(self, capacity, gamma, lam) -> None:
        self.states = [Batch(tmp=np.array([0]))] * capacity
        self.actions = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.state_values = np.zeros(capacity, dtype=np.float32)
        self.is_terminals = np.zeros(capacity, dtype=np.float32)
        self.next_states = [Batch(tmp=np.array([0]))] * capacity
        self.frames = [np.zeros((160, 256, 3), dtype=np.uint8)] * capacity

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.pointer, self.path_start_idx, self.max_capacity = 0, 0, capacity 
    
    def store(self, state, action, reward, log_prob, state_value, is_terminal, next_state, frame):
        
        assert self.pointer < self.max_capacity

        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.log_probs[self.pointer] = log_prob
        self.state_values[self.pointer] = state_value
        self.is_terminals[self.pointer] = is_terminal
        self.next_states[self.pointer] = next_state
        self.frames[self.pointer] = frame
        self.pointer += 1

    def calc_advantages(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.pointer)
        rewards = np.append(self.rewards[path_slice], last_val)
        values = np.append(self.state_values[path_slice], last_val)

        #GAE
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = discounted_cumsum(deltas, self.gamma*self.lam)

        # Rewards-to-go
        self.returns[path_slice] = discounted_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.pointer

    def get(self, device):
        
        assert self.pointer == self.max_capacity

        self.pointer, self.path_start_idx = 0, 0

        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / adv_std
        data = dict(actions = self.actions,
                    rewards = self.rewards,
                    log_probs = self.log_probs,
                    state_values = self.state_values,
                    is_terminals = self.is_terminals,
                    advantages = self.advantages,
                    returns = self.returns)
        
        data_dict = dict(states = Batch.cat(self.states), next_states = Batch.stack(self.next_states))
        tmp_dict = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        data_dict.update(tmp_dict)
        return PPODataset(data_dict, device)
    
    def make_video(self, path, reward):
        frames = self.frames[self.path_start_idx:self.pointer]
        video_path = path.joinpath(f"video_{reward}.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    
class Critic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None
        )
        self._device = device
    
    def forward(self, x):
        hidden = None
        return self.net(x), hidden

class ActorCritic(nn.Module):
    def __init__(self, cfg, device) -> None:
        self.cfg = cfg
        self.device = device
        super().__init__()

        #actor
        feature_net_kwargs = cfg.feature_net_kwargs
        feature_net = {}

        for k, v in feature_net_kwargs.items():

            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(F, cls)
            feature_net[k] = cls(**v, device=self.device)

        feature_fusion_kwargs = cfg.feature_fusion
        feature_net = SimpleFeatureFusion(
        feature_net, **feature_fusion_kwargs, device=self.device
        )

        self.actor = MultiCategoricalActor(
            feature_net,
            action_dim=[89],
            device=self.device,
            **cfg.actor,
        )

        self.agent = MineAgent(
            actor=self.actor,
        )

        self.critic = Critic(
            feature_net,
            output_dim=1,
            device=self.device,
            **cfg.actor
        )
    
    def act(self, state):
        logit_batch = self.agent(state)
        action = logit_batch.act
        action_logprob = logit_batch.dist.log_prob(action)
        
        state_feat, _ = self.actor.preprocess(state.obs)
        state_val, _ = self.critic(state_feat)

        return action, action_logprob, state_val
    
    def eval(self, state, action):

        logit_batch = self.agent(state)
        action_logprobs = logit_batch.dist.log_prob(action)
        action_probs = logit_batch.dist._dists[0].probs
        dist_entropy = logit_batch.dist.entropy()
        
        state_feat, _ = self.actor.preprocess(state.obs)
        state_values, _ = self.critic(state_feat)

        return action_logprobs, state_values, dist_entropy, action_probs
    
class PPO:
    def __init__(self, cfg, device) -> None:

        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.optim_iter = 15
        self.memory_capacity = 100000
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.cfg = cfg
        self.device = device

        self.buffer = PPOBuffer(self.memory_capacity, self.gamma, self.lam)
        self.policy = ActorCritic(cfg, device).to(device=self.device)
        self.optimizer = Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic},
        ])
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max= 10,
                                           eta_min = 5e-6)
        # self.actor_optim = Adam(self.policy.actor.parameters(), lr=self.lr_actor)
        # self.critic_optim = Adam(self.policy.critic.parameters(), lr=self.lr_critic)

    def process(self, state):
        with torch.no_grad():
            action, action_logprob, state_value = self.policy.act(state)
            return action, action_logprob, state_value

    def select_action(self, state):
        # with torch.no_grad():
        #     # state = torch.FloatTensor(state).to(self.device)
        #     action, action_logprob, state_val = self.policy.act(state)
        
        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.logprobs.append(action_logprob)
        # self.buffer.state_values.append(state_val)

        return self.process(state)[0]
    
    def plot_loss(self):
        fig, ax = plt.subplots((4,5))

        return 

    def KLD(self, P, Q):
        return (P * torch.log(P/Q)).sum(dim=1)
    
    def smoothing(self, probs, W):
        i = W - 1
        tmp = 0
        for j in range(1, W):
            tmp += self.KLD(probs[i:], probs[(i-j):(-j)])
        return tmp.mean()
    
    def save_model(self, res_path: pathlib.Path, time, epoch):
        weights_dir_path = res_path.joinpath("weights")
        if not weights_dir_path.exists():
            weights_dir_path.mkdir()
        
        weight_path = weights_dir_path.joinpath(f"model_{time}_{epoch}")
        torch.save(self.policy.state_dict(), weight_path)


    def update(self):
        dataset = self.buffer.get(self.device)
        dataloader = DataLoader(dataset, batch_size=4096, collate_fn=custom_collate_fn)
        # state_batch = data['states']
        # action_batch = data['actions']
        # advantage_batch = data['advantages']
        # old_logp_batch = data['log_probs']
        # return_batch = data['returns']

        for _ in range(self.optim_iter):
            print(f"Update {_}")
            for i, batch in enumerate(dataloader):
                state_batch, action_batch, advantage_batch, old_logp_batch, return_batch = batch

                logp_batch, value_batch, entropy_batch, probs_batch = self.policy.eval(state_batch, action_batch.unsqueeze(dim=0))
                
                # Define Loss function

                ratios = torch.exp(logp_batch - old_logp_batch)
                surr1 = ratios * advantage_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_batch
                
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = nn.MSELoss()(value_batch.squeeze(), return_batch)
                smoothing_loss = self.smoothing(probs_batch, 3)
                loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy_batch + 1e-7 * smoothing_loss
                print("loss:", loss.mean())
                wandb.log({"actor_loss": actor_loss.mean(), "critic_loss": critic_loss, "entropy_loss": entropy_batch, "smoothing_loss": smoothing_loss, "loss": loss.mean()})
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.policy.actor.parameters(), max_norm=10)
                clip_grad_norm_(self.policy.critic.parameters(), max_norm=10)
                self.optimizer.step()
        self.scheduler.step()

            # self.actor_optim.zero_grad()
            # actor_loss.backward(retain_graph=True)
            # self.actor_optim.step()
    
            # self.critic_optim.zero_grad()
            # critic_loss.backward()
            # self.critic_optim.step()

        # Optimaze policy 
        # for _ in range(self.K_epochs):
            # log_probs, state_values, dist_entropy = self.policy.eval(old_states, old_actions)
            # state_values = torch.squeeze(state_values)

            # ratios = torch.exp(log_probs - old_logprobs.detach())
            # surr1 = ratios * advantages
            # surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # actor_loss = (-torch.min(surr1, surr2)).mean()
            # critic_loss = nn.MSELoss()(state_values, rewards)

            # self.actor_optim.zero_grad()
            # actor_loss.backward(retain_graph=True)
            # self.actor_optim.step()

            # self.critic_optim.zero_grad()
            # critic_loss.backward()
            # self.critic_optim.step()