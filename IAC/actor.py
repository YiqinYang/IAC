import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os 
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ReparamGaussianPolicy(nn.Module):
    def __init__(self, args, agent_id, hidden_dim=64):
        super(ReparamGaussianPolicy, self).__init__()
        self.args = args
        self.max_action = args.high_action
        self.epsilon_D = 1e-6
        self.n_agents = self.args.n_agents

        self.actor_encoders = nn.ModuleList()
        for i in range(self.n_agents):
            self.encoder = nn.Sequential()
            self.encoder.add_module('fc1', nn.Linear(args.obs_shape[agent_id], hidden_dim))
            self.encoder.add_module('f1', nn.ReLU())
            self.encoder.add_module('fc2', nn.Linear(hidden_dim, hidden_dim))
            self.encoder.add_module('f2', nn.ReLU())
            self.encoder.add_module('mu', nn.Linear(hidden_dim, args.action_shape[agent_id]))
            self.actor_encoders.append(self.encoder)

        # self.mu_layer = nn.Linear(hidden_dim, args.action_shape[agent_id])
        self.num_k = self.args.n_agents * self.args.m
        
        self.all_encoder = nn.Sequential()
        self.all_encoder.add_module('all_fc1', nn.Linear(sum(args.obs_shape), hidden_dim))
        self.all_encoder.add_module('all_f1', nn.ReLU())
        self.all_encoder.add_module('all_fc2', nn.Linear(hidden_dim, hidden_dim))
        self.all_encoder.add_module('all_f2', nn.ReLU())
        self.K_out = nn.Linear(hidden_dim, self.num_k * args.action_shape[0])

    def forward(self, x, train=False, eval_mode=False):
        agents = range(self.args.n_agents) 
        inp = [torch.tensor(x[a_i], dtype=torch.float32).view(-1,self.args.obs_shape[0]) for a_i in agents]
        episode_num = inp[0].shape[0]
        
        mu = [self.max_action * torch.tanh(self.actor_encoders[a_i](inp[a_i])) for a_i in agents] # 网络不共享
        # mu = [self.max_action * torch.tanh(self.encoder(inp[a_i])) for a_i in agents] # 网络共享
        epsilons = [torch.tensor(np.random.rand(episode_num, self.args.action_shape[0]), dtype=torch.float) for _ in agents]
        locs = [(mu[a_i] + 0.05 * epsilons[a_i]).view(-1, self.args.action_shape[0]) for a_i in agents]
        
        if eval_mode:
            mu = torch.stack(mu, dim=0).view(-1, self.args.n_agents, self.args.action_shape[0])
            return mu
        else:
            if train:
                state = torch.stack(x, dim=1).view(episode_num, -1)
            else:
                state = torch.stack(inp, dim=1).view(episode_num, -1)
            s_all_encodings = self.all_encoder(state)

            # self.k_outs = self.K_out(s_all_encodings).view(episode_num, self.args.n_agents * self.args.action_shape[0], self.args.m)
            self.k_outs = (torch.tanh(self.K_out(s_all_encodings))).view(episode_num, self.args.n_agents * self.args.action_shape[0], self.args.m)
            diag = torch.ones(episode_num, self.args.n_agents * self.args.action_shape[0]) * self.epsilon_D
            locs = torch.stack(locs, dim=1).view(episode_num, self.args.n_agents * self.args.action_shape[0])
            dist = LowRankMultivariateNormal(locs, self.k_outs, diag)
            actions = dist.rsample()
            log_pi = dist.log_prob(actions)
            outs = actions.view(-1, self.args.n_agents, self.args.action_shape[0])

            return mu, outs, log_pi