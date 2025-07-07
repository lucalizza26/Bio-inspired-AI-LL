# IMPORTS
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import deque
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


HYPERPARAMS = {
    "layers": [],        # List defines architecture
    "epsilon_start": 1.0,
    "epsilon_end": 0.02,
    "epsilon_decay": 0.002,
    "gamma": 0.99,
    "lr": 1e-4,
    "batch_size": 64,
    "target_update_freq": 1000,
    "memory_size": 100_000,
    "min_replay_size": 1000
}

GAME_PARAMS = {
    "gravity": -10.0,            # Gravity in the environment
    "wind_power": 5.0,           # Wind power in the environment
    "turbulence_power": 0.25,     # Turbulence power in the environment
    "wind_enabled": True,       # Whether wind is enabled
}


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*samples))

    def __len__(self):
        return len(self.buffer)



def show_saved_model(model_path, episodes=5, render_mode="human", layers=[]):
    env = gym.make("LunarLander-v3", continuous=False, gravity=GAME_PARAMS["gravity"],
               enable_wind=GAME_PARAMS["wind_enabled"], wind_power=GAME_PARAMS["wind_power"],
                 turbulence_power=GAME_PARAMS["turbulence_power"], render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(obs_dim, n_actions, layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(state).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if render_mode == "human":
                env.render()
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
    env.close()



MODEL_FILEPATH = "saved_models/iteration_50_converged_937.pth"

show_saved_model(MODEL_FILEPATH, episodes=5, render_mode="human", layers=[256,128,64])