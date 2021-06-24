import numpy as np
import torch
import os
from IAC.actor import ReparamGaussianPolicy
from IAC.critics import AttentionCritic
from IAC.qmix_net import QMixNet

class Agent:
    def __init__(self, args):
        self.train_step = 0
        self.args = args
        self.alpha = 0.1
        self.actor = ReparamGaussianPolicy(args, 0)
        self.attn_critic = AttentionCritic(args)

        self.actor_target =ReparamGaussianPolicy(args, 0)
        self.target_attn_critic = AttentionCritic(args)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.target_attn_critic.load_state_dict(self.attn_critic.state_dict())

        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.critic_parameters = list(self.eval_qmix_net.parameters()) + list(self.attn_critic.parameters())
        self.critic_optim = torch.optim.Adam(self.critic_parameters, lr=self.args.lr_critic)

        # -------------------------------------------
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # if os.path.exists(self.model_path + '/149_actor_params.pkl'):
            # self.actor.load_state_dict(torch.load(self.model_path + '/149_actor_params.pkl'))
            # self.attn_critic.load_state_dict(torch.load(self.model_path + '/40_attn_critic_params.pkl'))
            # self.eval_qmix_net.load_state_dict(torch.load(self.model_path + '/40_mix_params.pkl'))
            # self.actor_target.load_state_dict(self.actor.state_dict())
            # self.target_attn_critic.load_state_dict(self.attn_critic.state_dict())
            # self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
            # print('Agent successfully loaded actor_network: {}'.format(self.model_path + '/actor_params.pkl'))

    def _hard_update_target_network(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.target_attn_critic.load_state_dict(self.attn_critic.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def select_action(self, o, eval_mode=False):
        if eval_mode:
            mu = self.actor(o, eval_mode=True)
            mu = mu.cpu().numpy().reshape(self.args.n_agents, self.args.action_shape[0])
            mu = np.clip(mu, -self.args.high_action, self.args.high_action)
            u = [mu[i] for i in range(self.args.n_agents)]
            return u.copy()
        else:
            _, pi, _ = self.actor(o)
            pi = pi.cpu().numpy().reshape(self.args.n_agents, self.args.action_shape[0])
            pi = np.clip(pi, -self.args.high_action, self.args.high_action)
            u = [pi[i] for i in range(self.args.n_agents)]
            return u.copy()

    def learn(self, transitions):
        agents = range(self.args.n_agents) 
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        o, u, o_next, r = [], [], [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
            r.append(transitions['r_%d' % agent_id])
        
        # ------------------------critic----------------------------------------
        with torch.no_grad():
            _, pi_next, next_log_pi = self.actor_target(o_next, train=True)
            u_next = [pi_next[:, a_i] for a_i in agents]
            target_critic_in = list(zip(o_next, u_next))
            q_next = self.target_attn_critic(target_critic_in)
            q_next_s = torch.stack(q_next, dim=-1)

        critic_in = list(zip(o, u))
        q = self.attn_critic(critic_in)
        q_s = torch.stack(q, dim=-1)
        # ------------------------critic----------------------------------------

        _, pi, log_pi = self.actor(o, train=True)
        action = [pi[:, a_i] for a_i in agents]
        policy_critic_in = list(zip(o, action))
        actor_qs = self.attn_critic(policy_critic_in)
        actor_qs = torch.stack(actor_qs, dim=-1)

        state = torch.stack(o, dim=1).view(self.args.batch_size, 1, -1)
        state_next = torch.stack(o_next, dim=1).view(self.args.batch_size, 1, -1)
     
        q_total_next = self.target_qmix_net(q_next_s, state_next).view(self.args.batch_size)
        r = torch.stack(r, dim=1).view(self.args.batch_size, -1)
        r = torch.mean(r, dim=-1)
        q_total_target = r + self.args.gamma * (q_total_next - self.alpha*next_log_pi)

        q_total_value = self.eval_qmix_net(q_s, state).view(self.args.batch_size)
        critic_loss = (q_total_target.detach() - q_total_value).pow(2).mean()

        actor_qs = torch.mean(actor_qs, dim=-1).view(self.args.batch_size)
        actor_loss = (self.alpha*log_pi - actor_qs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()  
        
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()

        if self.train_step % 200 == 0 and self.train_step > 1:
            self._hard_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        
        self.train_step += 1
        self.alpha = max(0.01, self.alpha - 0.000001)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.attn_critic.state_dict(),  model_path + '/' + num + '_attn_critic_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), model_path + '/' + num + '_mix_params.pkl')