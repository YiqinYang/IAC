import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from itertools import chain

class AttentionCritic(nn.Module):
    def __init__(self, args, hidden_dim=64, norm_in=False, attend_heads=4):
        super(AttentionCritic, self).__init__()
        self.args = args
        self.sa_size = args.sa_size
        self.n_agents = len(self.sa_size)
        self.attend_heads = attend_heads
        self.max_action = args.high_action

        sdim = args.obs_shape[0]
        adim = args.action_shape[0]
        idim = sdim + adim
        odim = adim

        self.critic_encoders = nn.ModuleList()
        self.critic_state_encoder = nn.ModuleList()
        self.critics = nn.ModuleList()

        # for i in range(self.n_agents):
        self.encoder = nn.Sequential()
        self.encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
        self.encoder.add_module('enc_nl', nn.ReLU())
            # self.critic_encoders.append(self.encoder)

        self.state_encoder = nn.Sequential()
        self.state_encoder.add_module('s_enc_fc1', nn.Linear(sdim, hidden_dim))
        self.state_encoder.add_module('s_enc_nl', nn.ReLU())
            # self.critic_state_encoder.append(self.state_encoder)

        self.critic = nn.Sequential()
        self.critic.add_module('critic_fc1', nn.Linear(2*hidden_dim, hidden_dim))
        self.critic.add_module('critic_nl_1', nn.ReLU())
        self.critic.add_module('critic_fc3', nn.Linear(hidden_dim, 1))
            # self.critics.append(self.critic)

        attend_dim = hidden_dim // attend_heads 
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.ReLU()))
        # self.shared_modules = [self.key_extractors, self.selector_extractors,
        #                        self.value_extractors, self.critic_encoders]
    
    def forward(self, inps, agents=None, return_q=True, regularize=False, return_attend=False):
        if agents is None:
            agents = range(self.args.n_agents) 
        states = [s for s, a in inps]
        inps = [torch.cat((s, a/self.max_action), dim=1) for s, a in inps]

        # s_encodings = [self.critic_state_encoder[a_i](states[a_i]) for a_i in agents] # no share
        s_encodings = [self.state_encoder(states[a_i]) for a_i in agents] # share
        # sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)] # no share
        sa_encodings = [self.encoder(inp) for inp in inps] # share
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]

        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]

                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))

                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1]) 
                
                attend_weights = F.softmax(scaled_attend_logits, dim=2)

                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2) 

                other_all_values[i].append(other_values) 
        
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            critic_in = torch.cat((sa_encodings[i], *other_all_values[i]), dim=1)
            # q = self.critics[a_i](critic_in) # no share
            q = self.critic(critic_in) # share

            if return_q:
                agent_rets.append(q)

            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)

        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

