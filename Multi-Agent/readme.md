# 多智能体强化学习

* [**MADDPG《Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments》**](https://github.com/PaperCommunity/Reinforcement-Learning/tree/master/Multi-Agent/Multi-Agent%20Actor-Critic%20for%20Mixed%20Cooperative-Competitive%20Environments)
  - **Update**: 2019.01.24
  - **Author**: ***
  - **Contact**: zhanghongming@pku.edu.cn
  - **Issues**:
  - **Overview**: 由于多智能体的环境状态由多个agent的行为共同决定，本身具有不稳定性(non-stationarity)，Q-learning算法很难训练，policy gradient算法的方差会随着智能体数目的增加变得更大。作者提出了一种actor-critic方法的变体MADDPG，对每个agent的强化学习都考虑其他agent的动作策略，进行中心化训练和非中心化执行，取得了显著效果。
