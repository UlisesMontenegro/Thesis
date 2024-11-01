# Importing dependencies
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

'''
# Defining rates & factors
lr = 0.01 # Learning Rate
gamma = 0.99 # Discount factor
total_episodes = 1000

# Defining Neural Network for the policy (Single one-layer MLP with 64 hidden units)
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # Set training mode
    
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x) # Forward pass
        pd = Categorical(logits=pdparam) # Probability Distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # Store for training
        return action.item()
    
def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # The returns
    future_ret = 0.0
    # Compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets # Gradient term; Negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward() # Backpropagate, compute gradients
    optimizer.step() # Gradient-ascent, update the weights (Consider Stochastic Gradient Descent/Ascent if the computational cost is very high)
    return loss

def main():
    # Creating environment
    env = gym.make("LunarLander-v2")
    in_dim = env.observation_space.shape[0] # 8
    out_dim = env.action_space.n # 4
    print("State space dimension: ", in_dim)
    print("Action space dimension: ", out_dim)
    pi = Pi(in_dim, out_dim) # Policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr)
    total_rewards = [] # Used to plot rewards per episode
    for episode in range(total_episodes):
        done = False
        state = env.reset()[0]
        episode_reward = 0
        while episode_reward < 200:
            action = pi.act(state)
            state, reward, truncated, terminated, _ = env.step(action)
            if truncated or terminated:
                done = True
            episode_reward += reward
            ("Episode Reward: ", episode_reward)
            if done:
                break
        loss = train(pi, optimizer) # Train per episode
        solved = episode_reward > 200
        pi.onpolicy_reset() # onpolicy: clear memory after training
        print(f'Episode {episode}, loss: {loss}, \
              total_reward: {episode_reward}, solved: {solved}')
        
        total_rewards.append(episode_reward)

    # Plotting rewards vs episodes
    plt.figure()
    plt.plot(np.arange(1, len(total_rewards)+1), total_rewards)
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.show()
'''

def main():
    env = gym.make("LunarLander-v2")

if __name__ == '__main__':
    main()

