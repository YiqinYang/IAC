import torch.nn as nn
import torch
import torch.nn.functional as F


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim) 
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  
        states = states.reshape(-1, self.args.state_shape)  

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  
        hidden = F.elu(torch.bmm(q_values, w1) + b1)  
        w2 = torch.abs(self.hyper_w2(states))  
        b2 = self.hyper_b2(states)  

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
