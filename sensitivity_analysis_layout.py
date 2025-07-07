import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import deque
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from trainingV4 import HYPERPARAMS, GAME_PARAMS, QNetwork, ReplayBuffer, train_dqn


# find layer layout permutations:
layouts = []

layouts.append([64, 256, 256])

for i in [128, 256]:
    layouts.append([i])
    for j in [64, 128, 256]:
        layouts.append([i, j])
        for k in [64, 128, 256]:
                layouts.append([i, j, k])
 



for layout in layouts:
    HYPERPARAMS["layers"] = layout
    print(f"Testing layout: {layout}")
    CSV_FILENAME = f"layout_analysis/lunar_lander_{'_'.join(map(str, layout))}.csv"

    env = gym.make("LunarLander-v3", continuous=False, gravity=GAME_PARAMS["gravity"],
               enable_wind=GAME_PARAMS["wind_enabled"], wind_power=GAME_PARAMS["wind_power"],
                turbulence_power=GAME_PARAMS["turbulence_power"]) 
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = QNetwork(obs_dim, n_actions, HYPERPARAMS["layers"]).to(device)
    target_net = QNetwork(obs_dim, n_actions, HYPERPARAMS["layers"]).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=HYPERPARAMS["lr"])
    replay_buffer = ReplayBuffer(HYPERPARAMS["memory_size"])
    log_dir = "C:/Users/lucal/Desktop/tb_logs"
    writer = SummaryWriter(log_dir=log_dir)
    # writer = SummaryWriter()

    obs, _ = env.reset()
    for _ in range(HYPERPARAMS["min_replay_size"]):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push((obs, action, reward, next_obs, done))
        obs = next_obs if not done else env.reset()[0]

    obs, _ = env.reset()
    episode_reward = 0
    episode_num = 0



    train_dqn(env, policy_net, target_net, optimizer, replay_buffer, writer,
            HYPERPARAMS["batch_size"], HYPERPARAMS["gamma"], HYPERPARAMS["target_update_freq"],
                saving_freq=100, csv_filename=CSV_FILENAME, max_episodes=1000)