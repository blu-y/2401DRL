import gym
import numpy as np
import random
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

### Action Space
# 0:Left, 1:Down, 2:Right, 3:Up'
actions = {0: '←', 1: '↓', 2: '→', 3: '↑'}
m = map_size = 3
epsilon = 1 # epsilon
sr = 0.8 # success rate
gamma = 0.9 # discount factor
alpha = 0.2 # learning rate

def sind_to_mapind(i, m):
    return i%m, m-i//m-1

def plot(env, states, Q, action=-1):
    k = len(Q) - 1
    step = len(states) - 1
    state = states[-1][0]
    plt.figure(1, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.cla()
    if step == 0: 
        plt.title("k: %d, Step: %d State: %d" % (k, step, state))
    else: plt.title("k: %d, Step: %d Action: %s State: %d" % (k, step, actions[action], state))
    plt.imshow(env.render())
    plt.axis('off')

    q = Q[-1]
    plt.figure(1, figsize=(3*m, 2*m))
    plt.subplot(1, 2, 2)
    plt.cla()
    for i, _s in enumerate(q):
        _i, _j = sind_to_mapind(i, m)
        plt.text(_i*3+1, _j*3+2, round(_s[0], 3), ha='center', va='center', fontsize=3*m)
        plt.text(_i*3+2, _j*3+1, round(_s[1], 3), ha='center', va='center', fontsize=3*m)
        plt.text(_i*3+3, _j*3+2, round(_s[2], 3), ha='center', va='center', fontsize=3*m)
        plt.text(_i*3+2, _j*3+3, round(_s[3], 3), ha='center', va='center', fontsize=3*m)
    plt.axis('off')
    plt.grid(True)
    plt.xlim(0, 3*m)
    plt.ylim(0, 3*m)
    plt.show(block=False)
    plt.pause(0.15)

def sample_action(Q, state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[-1][state[0]])
    
def sample_state(state, action):
    keys = [*actions.keys()]
    keys.remove(action)
    if random.random() > sr:
        return env.step(action)
    else: return env.step(random.choice(keys))

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size), is_slippery=False, render_mode='rgb_array')
ms = m**2
n = env.action_space.n
# Initialize Q-table
Q = [np.zeros((map_size**2, n))]
reset = True
action = []
while True:
    # Get initial state
    if reset:
        states = [env.reset()]
        plot(env, states, Q)
        reset = False
    # Sample action a (e-greedy)
    action.append(sample_action(Q, states[-1][0]))
    # Sample next state s' (Non slippery)
    state = sample_state(states[-1], action[-1])
    states.append(state)
    plot(env, states, Q, action[-1])
    # Check s' is terminal
    target = states[-1][1]
    if states[-1][2]:
        reset = True
    else: 
        target += gamma * Q[-1][state[0]][np.argmax(Q[-1][state[0]])]
    Q.append(Q[-1].copy())
    Q[-1][states[-2][0]][action[-1]] = (1-alpha)*Q[-2][states[-2][0]][action[-1]] + alpha*target
    print(Q[-1]-Q[-2])

plt.pause(5)