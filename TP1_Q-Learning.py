### Terminal Project 1: Q-Learning
# 2023320001 윤준영
# python 3.8.3 / gym 0.26.2
# numpy 1.24.4 / matplotlib 3.7.5 / pygame = 2.5.2

import gym
import numpy as np
import random
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


###################PARAMETERS####################
m = map_size = 4 # m x m map size               #
epsilon = 0.3 # epsilon (0: greedy, 1 random)   #
sr = 0.8 # success rate                         #
gamma = 0.9 # discount factor                   #
alpha = 0.3 # learning rate                     #
render = True # to see plot                     #
#################################################


 
def sample_action(Q, state):
    ### SAMPLE ACTION (e-greedy)
    # epsilon : to random action
    # 1-epsilon : to maximize Q value action
    if random.random() < epsilon:
        return env.action_space.sample()
    else: 
        # to deal with multiple max values
        maxind = np.where(Q[state[0]] == np.max(Q[state[0]]))[0].tolist()
        return random.sample(maxind, 1)[0]
    
def sample_state(state, action):
    ### SAMPLE NEXT STATE
    # sr(success rate): to move to the action,
    # 1-sr: move to the other random directions
    if random.random() > sr:
        keys = [*actions.keys()]
        keys.remove(action)
        action = random.choice(keys)
    return env.step(action), action

def sind_to_mapind(i, m):
    ### STATE INDEX TO MAP INDEX(plt) for plotting
    return i%m, m-i//m-1

def plot(env, k, step, state, Q, action=-1):
    ### PLOT ENVIRONMENT AND Q VALUES
    if not render: return
    
    # plot environment
    state = state[0]
    plt.figure(1, figsize=(3.5*m, 1.5*m))
    plt.subplot(1, 2, 1)
    plt.cla()
    str = "k: %d, Step: %d, State: %s" % (k, step, state)
    if action >= 0: str += ", Action: %s" % actions[action]
    plt.title(str)
    plt.axis('off')
    plt.imshow(env.render())

    # plot Q values
    plt.figure(1, figsize=(3*m, 2*m))
    plt.subplot(1, 2, 2)
    plt.cla()
    for i, _s in enumerate(Q):
        _i, _j = sind_to_mapind(i, m)
        plt.text(_i*3+1, _j*3+2, round(_s[0], 3), ha='center', va='center', fontsize=2.2*m)
        plt.text(_i*3+2, _j*3+1, round(_s[1], 3), ha='center', va='center', fontsize=2.2*m)
        plt.text(_i*3+3, _j*3+2, round(_s[2], 3), ha='center', va='center', fontsize=2.2*m)
        plt.text(_i*3+2, _j*3+3, round(_s[3], 3), ha='center', va='center', fontsize=2.2*m)
    plt.axis('off')
    plt.xlim(0, 3*m)
    plt.ylim(0, 3*m)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size),
               is_slippery=False, render_mode='rgb_array')
# 0:Left, 1:Down, 2:Right, 3:Up'
actions = {0: '←', 1: '↓', 2: '→', 3: '↑'}
n = env.action_space.n
# Initialize Q-table
Q = np.zeros((map_size**2, n))
Q_ = Q.copy()
k = 0
step = 0
success = 0
while True:
    ### Get initial state
    if step == 0:
        ### check convergence
        s = env.reset()
        s = [s[0], 0.0, False, False, s[-1]]
        plot(env, k, step, s, Q)
    ### Sample action a (e-greedy)
    a = sample_action(Q, s)
    ### Sample next state s' (Non slippery)
    s_, a = sample_state(s, a)
    ### Check s' is terminal
    # if terminal state, target is reward
    # if not terminal, target is reward + gamma*max(Q(s',a'))
    if s_[2]:
        # check success, reset step, increase k
        if s_[1] == 1: 
            success += 1
            if success/(k+1) > 0.1 and not (Q==Q_).all() and np.sum(abs(Q-Q_)) < 1e-5: 
                # print(success/(k+1), not (Q==Q_).all(), np.sum(abs(Q-Q)))
                break
        target = s_[1]
        step = 0
        k += 1
        Q = Q_.copy()
    else: 
        target = s[1] + gamma*np.max(Q[s_[0]])
        step += 1
    ### Update Q table
    # Q(s,a) = (1-alpha)*Q(s,a) + alpha*target
    # if actually moved (to avoid wall)
    if s[0] != s_[0]:
        Q_[s[0]][a] = (1-alpha)*Q[s[0]][a] + alpha*target
        if target > 0:
            print('k=%d, step=%d\tQ\'(%d,%d): %.3f -> %.3f'
                  %(k, step, s[0], a, Q[s[0]][a], Q_[s[0]][a]))
    # set s <- s'
    s = s_
    plot(env, k, step, s_, Q, a)

plot(env, k, step, s_, Q_, a)
print(Q_)
print('Converged!')
plt.show(block=True)