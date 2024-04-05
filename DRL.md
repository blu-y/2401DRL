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

-> to find policy \pi

과제에서 state: x, y 좌표

```python
class Action(Enum):
	UP = 0
	def move(self...)
		...
```

#### Grid World

state-action-transition probabilities: array shape = (xdim, ydim, #actions, #next-states)

probability[3, 2, 0, 0] = 0.8	

(broadcast : dimension matching, ex) new_axis, squeeze, ...)

prob bias 

Q-Values - Bellman Equation

    V(s), Q(s, a)

#### Policy Iteration

Policy Evaluation : policy is given(Value iteration에서는 확률로 주어졌었음)

Stochastic Policy : 2

#### Maximum Entropy MDP

Entropy : measure of uncertainty over random variable X

Lagrangian multiple








## Policy Gradient
#### Likelihood Ratio Gradient Estimate
no bias : original과 estimation의 expectation이 같다.
#### Baseline Substraction
b가 theta의 함수가 아니면,,
#### Baseline Choices
-> state-dependent expected return