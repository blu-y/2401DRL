### Lay the Foundations for Deep Reinforcement Learning
1. Understand basic theories.
2. Implement the theories and analyze the experiments.

### Fundamental Theory
1. MDP
2. DQL
3. Policy Gradient
4. PPO
5. DDPG
6. Model Based RL

### MDP
#### Optimal Value Function V*
= sum of discounted rewards when starting from state s and **acting optimally**
V*(4,3) = 1
discounted: gamma
- w/p = 0.8, r = 0.9, H = 100
    : V*(4,3) = 1
      V*(3,3) = 0.8 * 0.9 * V*(4,3)
                + 0.1 * 0.9 * V*(3,3)
                + 0.1 * 0.9 * V*(3,2)
-> set initial value and iterate
    : dynamic programming
#### Value Iteration
**value update** or **Bellman update/back-up**