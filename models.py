import numpy as np
import matplotlib.pyplot as plt

import pathlib
import logging
import wandb

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mineclip.mineagent import features as F
from mineclip import SimpleFeatureFusion, MineAgent, MultiCategoricalActor
from mineclip.mineagent.batch import Batch
from mineclip.utils import build_mlp

from Data.buffers import PPOBuffer, SIBuffer
from Data.datasets import custom_collate_fn
import pdb
    
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

class Actor(nn.Module):
    def __init__(self, cfg, device) -> None:
        self.cfg = cfg
        self.device = device
        super().__init__()

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


class ActorCritic(nn.Module):
    def __init__(self, cfg, device) -> None:
        self.cfg = cfg
        self.device = device
        super().__init__()

        self.actor = Actor(
            cfg=self.cfg,
            device=self.device
        )

        self.critic = Critic(
            self.actor.actor.preprocess,
            output_dim=1,
            device=self.device,
            **cfg.actor
        )
    
    def act(self, state):
        logit_batch = self.actor.agent(state)
        action = logit_batch.act
        action_logprob = logit_batch.dist.log_prob(action)
        
        state_feat, _ = self.actor.actor.preprocess(state.obs)
        state_val, _ = self.critic(state_feat)

        return action, action_logprob, state_val
    
    def eval(self, state, action):
        logit_batch = self.actor.agent(state)
        action_logprobs = logit_batch.dist.log_prob(action)
        action_probs = logit_batch.dist._dists[0].probs
        dist_entropy = logit_batch.dist.entropy()
        
        state_feat, _ = self.actor.actor.preprocess(state.obs)
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
        self.counter = 0

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
    
    def save_model(self, res_path: pathlib.Path, epoch):
        weights_dir_path = res_path.joinpath("weights")
        if not weights_dir_path.exists():
            weights_dir_path.mkdir()
        
        weight_path = weights_dir_path.joinpath(f"model_epoch_{epoch+1}")
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
            logging.info(f"PPO update {_+1}")
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
                logging.info(f"iter: {i+1}, loss: {loss.mean()}")
                self.counter += 1
                # wandb.log({
                #     "actor_loss": actor_loss.mean(), 
                #     "critic_loss": critic_loss, 
                #     "entropy_loss": entropy_batch, 
                #     "smoothing_loss": smoothing_loss, 
                #     "loss": loss.mean(),
                #     "update": self.counter,
                # })
                
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

class SImodel:
    def __init__(self, cfg, device) -> None:
        self.epochs = 10
        self.lr = 1e-4
        self.buffer_capacity = 100
        self.cfg = cfg
        self.device = device
        self.counter = 0

        self.buffer = SIBuffer(self.buffer_capacity)
        self.model = Actor(self.cfg, self.device).to(device=self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max= 10,
                                           eta_min = 1e-6)

    def eval(self, state, action):
        logit_batch = self.model.agent(state)
        action_logprobs = logit_batch.dist.log_prob(action)
        return action_logprobs

    def train(self, weights):
        trajectories = self.buffer.sample()
        data_batch = Batch.cat(trajectories)
        state_batch = Batch(obs=data_batch.obs)
        action_batch = data_batch.actions.to(self.device)
        # state_batch = data_batch.obs
        self.model.load_state_dict(weights)

        for epoch in range(self.epochs):
            loss = - self.eval(state_batch, action_batch.unsqueeze(dim=0)).mean() #maximize E(logp(a|s))
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.optimizer.step()
            self.counter += 1
            logging.info(f"SI update {epoch+1}, loss: {loss}")
            # wandb.log({'SI_loss': loss, 'epoch': self.counter,})
        self.scheduler.step()
