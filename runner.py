from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = Agent(self.args)
        self.buffer = Buffer(args)
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            with torch.no_grad():
                actions = self.agents.select_action(s)
            # self.env.render()
            # time.sleep(0.05)
            u = actions
            for i in range(self.args.n_agents, self.args.n_players):  # i = 3
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                self.agents.learn(transitions)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                with open('IAC_episode_rewards.txt','w') as f: 
                    for i in returns:
                        f.write(str(i) + ' ')
                        f.write('\n')
                    f.close()                
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                # time.sleep(0.05)
                with torch.no_grad():
                    actions = self.agents.select_action(s, eval_mode=True)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes