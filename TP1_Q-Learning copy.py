import gym
import numpy as np
import random
from gym.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# python 3.8.3 / gym 0.26.2 / numpy 1.24.4 / matplotlib 3.7.5

###################PARAMETERS####################
m = map_size = 4                                #
epsilon = 0.3 # epsilon (0: greedy, 1 random)   #
sr = 0.8 # success rate                         #
gamma = 0.9 # discount factor                   #
alpha = 0.3 # learning rate                     #
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
    state = state[0]
    plt.figure(1, figsize=(3*m, 1.5*m))
    plt.subplot(1, 2, 1)
    plt.cla()
    str = "k: %d, Step: %d, State: %s" % (k, step, state)
    if action >= 0: str += ", Action: %s" % actions[action]
    plt.title(str)
    plt.imshow(env.render())
    plt.axis('off')

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
    plt.grid(True)
    plt.xlim(0, 3*m)
    plt.ylim(0, 3*m)
    plt.show(block=False)
    plt.pause(0.01)

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size), is_slippery=False, render_mode='rgb_array')
# 0:Left, 1:Down, 2:Right, 3:Up'
actions = {0: '←', 1: '↓', 2: '→', 3: '↑'}
n = env.action_space.n
# Initialize Q-table
Q = np.zeros((map_size**2, n))
_Q = Q.copy()
k = 0
step = 0
success = 0
while True:
    ### Get initial state
    if step == 0:
        ### check convergence
        if success/(k+1) > 0.1 and not (Q==_Q).all() and np.sum(abs(_Q-Q)) < 1e-5: 
            # print(success/(k+1), not (Q==_Q).all(), abs(np.sum(_Q-Q)))
            break
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
        # check success, reset step, increase k, _Q <- Q
        if s_[1] == 1: success += 1
        target = s_[1]
        step = 0
        k += 1
        _Q = Q.copy()
    else: 
        target = s[1] + gamma*np.max(Q[s_[0]])
        step += 1
    print('k: %d, step: %d' %(k, step))
    ### Update Q table
    # Q(s,a) = (1-alpha)*Q(s,a) + alpha*target
    # if actually moved (to avoid wall)
    if s[0] != s_[0]:
        Q[s[0]][a] = (1-alpha)*_Q[s[0]][a] + alpha*target
        print('\tQ(%d, %d) updated: %.3f to %.3f' %(s[0], a, _Q[s[0]][a], Q[s[0]][a]))
    # set s <- s'
    s = s_
    plot(env, k, step, s_, Q, a)
print('Converged!')
plt.show(block=True)