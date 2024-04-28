from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch import optim
from tqdm import tqdm

import gymnasium as gym
        
# environment hyperparams
n_envs = 16
n_updates = 60000
n_heads = 17
save_weights = True
load_weights = True
n_showcase_episodes = 3

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x
   
class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, num_heads, hidden_dim, dropout_rate):
        super(Actor, self).__init__()
        self.transformer = TransformerBlock(input_dim, num_heads, hidden_dim, dropout_rate)
        self.mean_head = nn.Sequential(
            nn.Linear(num_heads, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(num_heads, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(0)  # Add a batch dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove the batch dimension
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp for numerical stability
        return mean, log_std.exp()

    def sample_action(self, state):
        mean, std = self(state)
        normal_dist = Normal(mean, std)
        action = normal_dist.rsample()  # Differentiable sampling
        log_prob = normal_dist.log_prob(action)
        entropy = normal_dist.entropy()
        return action, log_prob, entropy
    
class Critic(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_rate):
        super(Critic, self).__init__()
        self.transformer = TransformerBlock(input_dim, num_heads, hidden_dim, dropout_rate)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add a batch dimension
        x = self.transformer(x)
        x = x.squeeze(0)  # Remove the batch dimension
        value = self.value_head(x)
        return value 

class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int, # 17개
        n_actions: int, # 6개
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        n_envs: int,
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        hidden_dim = 256
        ##### n_heads = 17 
        dropout_rate = 0.1
        self.actor = Actor(n_features, n_actions, n_heads, hidden_dim, dropout_rate).to(self.device)
        self.critic = Critic(n_features, n_heads, hidden_dim, dropout_rate).to(self.device)

        # define optimizers for actor and critic
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=actor_lr)

    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        x = torch.Tensor(x).to(self.device)
        x = x.unsqueeze(0)
        x1 = self.transformer_block1(x)
        action_logits_vec = self.actor(x1)  # shape: [n_envs, n_actions]
        x2 = self.transformer_block2(x)
        state_values = self.critic(x2)  # shape: [n_envs,]
        return (state_values, action_logits_vec)

    def select_action(
        self, x: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)
        actions, action_log_probs, entropy = self.actor.sample_action(x)
        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        n_steps: int,
        # lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = len(rewards)
        n_action = action_log_probs.shape[1]
        returns = torch.zeros(T, n_action, self.n_envs, device=device)
        advantages = torch.zeros_like(returns)
        # compute the returns using n-step returns
        for t in range(T - n_steps):
            n_step_return = value_preds[t + n_steps] * (gamma ** n_steps)
            for step in reversed(range(n_steps)):
                n_step_return = rewards[t + step] + gamma * n_step_return * masks[t + step]
            returns[t] = n_step_return
            advantages[t] = returns[t] - value_preds[t]
        # calculate the critic loss
        critic_loss = advantages.pow(2).mean()
        # calculate the actor loss, using the advantages and including entropy bonus for exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return critic_loss, actor_loss
    
        # T = len(rewards)
        # n_action = action_log_probs.shape[1]
        # advantages = torch.zeros(T, n_action, self.n_envs, device=device)
        # # compute the advantages using GAE
        # gae = 0.0
        # for t in reversed(range(T - 1)):
        #     td_error = (
        #         rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
        #     )
        #     gae = td_error + gamma * lam * masks[t] * gae
        #     advantages[t] = gae
        # # calculate the loss of the minibatch for actor and critic
        # critic_loss = advantages.pow(2).mean()
        # # give a bonus for higher entropy to encourage exploration
        # actor_loss = (
        #     -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        # )
        # return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
n_steps_per_update = 128
randomize_domain = True

# agent hyperparams
gamma = 0.999
# lam = 0.95  # hyperparameter for 
n_steps = 32 
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
actor_lr = 0.001
critic_lr = 0.005

# Note: the actor has a slower learning rate so that the value targets become
# more stationary and are theirfore easier to estimate for the critic

# environment setup
if randomize_domain:
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                "Walker2d-v4",
                forward_reward_weight=np.clip(
                    np.random.normal(loc=1.0, scale=1.0), 0.5, 1.5
                ),
                ctrl_cost_weight=np.clip(
                    np.random.normal(loc=1e-3, scale=1e-3), 5e-4, 1.5e-3
                ),
                healthy_reward=np.clip(
                    np.random.normal(loc=1.0, scale=1.0), 0.5, 1.5
                ),
                reset_noise_scale=np.clip(
                    np.random.normal(loc=5e-3, scale=5e-3), 1e-4, 1.5e-3
                ),
                # exclude_current_positions_from_observation=False,
                max_episode_steps=10000,
            )
            for i in range(n_envs)
        ]
    )

else:
    envs = gym.vector.make("Walker2d-v4", num_envs=n_envs, max_episode_steps=10000)


obs_shape = envs.single_observation_space.shape[0]
action_shape = envs.single_action_space.shape[0]

# set the device
use_cuda = False
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# init the agent
print(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)
agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

# create a wrapper environment to save episode returns and episode lengths
envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

critic_losses = []
actor_losses = []
entropies = []

# use tqdm to get a progress bar for training
for sample_phase in tqdm(range(n_updates)):
    # we don't have to reset the envs, they just continue playing
    # until the episode is over and then reset automatically

    # reset lists that collect experiences of an episode (sample phase)
    ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_action_log_probs = torch.zeros(n_steps_per_update, action_shape, n_envs, device=device)
    masks = torch.zeros(n_steps_per_update, n_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        states, info = envs_wrapper.reset(seed=42)

    # play n steps in our parallel environments to collect data
    for step in range(n_steps_per_update):
        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy = agent.select_action(
            states
        )
        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = envs_wrapper.step(
            actions.cpu().detach().numpy()
        )

        ep_value_preds[step] = torch.squeeze(state_value_preds)
        ep_rewards[step] = torch.tensor(rewards, device=device)
        ep_action_log_probs[step] = action_log_probs.squeeze().transpose(0,1)

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
        masks[step] = torch.tensor([not term for term in terminated])

    # calculate the losses for actor and critic
    critic_loss, actor_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        entropy,
        masks,
        gamma,
        n_steps,
        ent_coef,
        device,
    )

    # update the actor and critic networks
    agent.update_parameters(critic_loss, actor_loss)

    # log the losses and entropy
    critic_losses.append(critic_loss.detach().cpu().numpy())
    actor_losses.append(actor_loss.detach().cpu().numpy())
    entropies.append(entropy.detach().mean().cpu().numpy())

""" plot the results """

# %matplotlib inline

rolling_length = 20
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
fig.suptitle(
    f"Training plots for {agent.__class__.__name__} in the Walker2d-v4 environment \n \
             (n_envs={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
)

# episode return
axs[0][0].set_title("Episode Returns")
episode_returns_moving_average = (
    np.convolve(
        np.array(envs_wrapper.return_queue).flatten(),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[0][0].plot(
    np.arange(len(episode_returns_moving_average)) / n_envs,
    episode_returns_moving_average,
)
axs[0][0].set_xlabel("Number of episodes")

# entropy
axs[1][0].set_title("Entropy")
entropy_moving_average = (
    np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][0].plot(entropy_moving_average)
axs[1][0].set_xlabel("Number of updates")


# critic loss
axs[0][1].set_title("Critic Loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0][1].plot(critic_losses_moving_average)
axs[0][1].set_xlabel("Number of updates")


# actor loss
axs[1][1].set_title("Actor Loss")
actor_losses_moving_average = (
    np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][1].plot(actor_losses_moving_average)
axs[1][1].set_xlabel("Number of updates")

plt.tight_layout()
plt.show()


actor_weights_path = "weights/actor_weights.h5"
critic_weights_path = "weights/critic_weights.h5"

if not os.path.exists("weights"):
    os.mkdir("weights")

""" save network weights """
if save_weights:
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)


""" load network weights """
if load_weights:
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)
    agent.actor.load_state_dict(torch.load(actor_weights_path))
    agent.critic.load_state_dict(torch.load(critic_weights_path))
    agent.actor.eval()
    agent.critic.eval()

""" play a couple of showcase episodes """


for episode in range(n_showcase_episodes):
    print(f"starting episode {episode}...")
    env = gym.make("Walker2d-v4", render_mode="human", 
                #    exclude_current_positions_from_observation=False,
                   terminate_when_unhealthy=False,
                   max_episode_steps=10000)

    # get an initial state
    state, info = env.reset()

    # play one episode
    done = False
    while not done:
        # select an action A_{t} using S_{t} as input for the agent
        with torch.no_grad():
            action, _, _, _ = agent.select_action(state[None, :])

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step(action.cpu().numpy().squeeze())

        # update if the environment is done
        done = terminated or truncated
env.close()

















