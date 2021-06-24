# IAC-AAMAS(2021)

This is a pytorch implementation of IAC on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of IAC is [Modeling the Interaction between Agents in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2102.06042).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Quick Start

```shell
$ python3 main.py --scenario-name=simple_tag --evaluate-episodes=10
```

Directly run the main.py, then the algrithm will be trained on scenario 'simple_tag' for 10 episodes.

## Note

+ The POMDP version of simple_tag is in POMDP_tag.py. You can put it in the MPE environment and enjoy it.

+ There are 4 agents in simple_tag, including 3 predators and 1 prey. we use IAC to train predators to catch the prey. The prey's action can be controlled by you, in our case we set it random. 

+ The default setting of Multi-Agent Particle Environment(MPE) is sparse reward, you can change it to dense reward by replacing 'shape=False' to 'shape=True' in file multiagent-particle-envs/multiagent/scenarios/simple_tag.py/.

+ Our work is basic, and I think someone can explore some exciting directions based on this work. If you have any questions, please contact me: yangyiqi19@mails.tsinghua.edu.cn.
