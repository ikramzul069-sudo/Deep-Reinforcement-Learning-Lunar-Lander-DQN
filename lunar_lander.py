import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Neural Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # 2 Hidden Layers with 64 unit (Standard LunarLander)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 2. DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.99            # Discount factor
        self.TAU = 1e-3              # Soft update parameter
        self.lr = 5e-4               # Learning rate
        self.batch_size = 64         # batch training
        self.update_every = 4        # Learn every 4 steps
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay Memory
        self.memory = deque(maxlen=100000)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.t_step = 0 # Counter for update_every

        self.losses = []

    def step(self, state, action, reward, next_state, done):
        # Save experience to Replay Buffer
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        # learn each update step
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()

    def act(self, state, eps=0.0):
        # Epsilon-Greedy Action Selection
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Exploration and Exploutation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to Tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        # Get max predicted Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets (Bellman Equation)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Loss & Backprop
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft Update
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)

        # Save losses
        self.losses.append(loss.item())

# 3. Main Loop for Training
def train_dqn():
    env = gym.make('LunarLander-v3') #render_mode= "human"
    agent = DQNAgent(state_size=8, action_size=4)
    
    n_episodes = 1000
    all_scores = []
    all_losses = agent.losses

    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    scores_window = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        all_scores.append(score)
        eps = max(eps_end, eps_decay * eps) # Decay epsilon
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
               
        #Save the model
        torch.save(agent.qnetwork_local.state_dict(), 'lunar_lander_model.pth')
    
    # Plotting rewards and losses
    #plot_results(all_scores, all_losses)

    env.close()

def plot_results(scores, losses):
    plt.figure(figsize=(12, 5))
    
    # Plot Rewards
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Learning Progress (Rewards)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('plot.png')

def record_final_agent():
    # Record Video
    env = gym.make('LunarLander-v3', render_mode="rgb_array") 
    env = gym.wrappers.RecordVideo(env, video_folder="./videos", name_prefix="final_landing")
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Call agent and load training result
    agent = DQNAgent(state_size, action_size)
    agent.qnetwork_local.load_state_dict(torch.load('lunar_lander_model.pth'))
    agent.qnetwork_local.eval() # Set to evaluation mode

    state, _ = env.reset()
    done = False
    total_reward = 0

    print("Last Landing")
    
    while not done:
        env.render()
        
        # Best action
        action = agent.act(state, eps=0.0) 
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Finish Landing! Total Score: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_dqn()
    #record_final_agent()