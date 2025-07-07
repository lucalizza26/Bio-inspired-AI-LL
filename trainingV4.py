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



# Setup
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

def train_dqn(
    env,
    policy_net,
    target_net,
    optimizer,
    replay_buffer,
    writer,
    batch_size,
    gamma,
    target_update_freq,
    saving_freq,
    csv_filename,
    max_episodes,
    iteration = None
):
    episode_num = 0
    frame_idx = 0
    with open(csv_filename, 'w') as f:
        f.write("")

    training = True

    while training: # episode loop
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done: # frame loop
            frame_idx += 1
            epsilon = epsilon_linear(episode_num)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy_net.net[0].weight.device)
                    action = policy_net(state).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward

            # Sample minibatch and train
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)

            obs_batch = torch.tensor(obs_batch, dtype=torch.float32).to(policy_net.net[0].weight.device)
            act_batch = torch.tensor(act_batch, dtype=torch.int64).to(policy_net.net[0].weight.device)
            rew_batch = torch.tensor(rew_batch, dtype=torch.float32).to(policy_net.net[0].weight.device)
            next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32).to(policy_net.net[0].weight.device)
            done_batch = torch.tensor(done_batch, dtype=torch.float32).to(policy_net.net[0].weight.device)

            q_values = policy_net(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_obs_batch).max(1)[0]
                target_q = rew_batch + gamma * next_q_values * (1 - done_batch)

            loss = nn.MSELoss()(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Loss", loss.item(), frame_idx)
            writer.add_scalar("Epsilon", epsilon, frame_idx)

            if frame_idx % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        save_line(csv_filename, episode_num, episode_reward)
        writer.add_scalar("Reward", episode_reward, episode_num)
        to_print = f"Episode: {episode_num}, Reward: {int(episode_reward)}, Epsilon: {epsilon:.4f}"

        # if saving_freq != 0:
        #     if episode_num % saving_freq == 0:
        #         torch.save(policy_net.state_dict(), f"saved_models\\checkpoint_{episode_num}_{'_'.join(map(str, HYPERPARAMS['layers']))}.pth")

        if convergence_check(csv_filename, to_print=to_print):
            training = False
            print(f"Convergence achieved at episode {episode_num}.")
            torch.save(policy_net.state_dict(), f"saved_models\\iteration_{iteration}_converged_{episode_num}.pth")

        episode_num += 1
        if episode_num >= max_episodes:
            training = False
    

# def convergence_check(csv_filename, window=100, threshold=200, to_print=None):
#     rewards = []
#     with open(csv_filename, 'r') as f:
#         for line in f:
#             ep, reward = line.strip().split(',')
#             rewards.append((int(ep), float(reward)))
#     episode_nums, rewards = zip(*rewards)
#     rewards_np = np.array(rewards)
#     condition_a = np.sum(rewards_np[-window:] > threshold)
#     condition_b = np.mean(rewards_np[-2*window:])
#     print(f"{to_print} -- {condition_a} successful out of last {window}, Avergage over last {2*window}: {condition_b}")
#     if condition_a > 95 and condition_b > 180:
#         return True
#     else:
#         return False

# OLD VERSION OF COVERGENCE CHECK FUNCTION
    

def convergence_check(csv_filename, to_print=None):
    eps = []
    rs = []
    with open(csv_filename, 'r') as f:
        for line in f:
            ep, reward = line.strip().split(',')
            eps.append(ep)
            rs.append(reward)

    for i in range(len(eps)):
        if i >= 100:
            successes = np.sum(np.array(rs[max(0, i-100):i]) > 200)
            std = np.std(rs[max(0, i-100):i])
        else:
            successes = np.sum(np.array(rs[:i]) > 200)
            std = np.std(rs[:i])
        if i >= 200:
            mean = np.mean(rs[max(0, i-200):i])
        else:
            mean = np.mean(rs[:i]) if i > 0 else 0
        print(f"{to_print} -- {successes} successful out of last 100, Avergage over last 200: {mean}")
        if successes >= 95 and mean >= 180:
            return True
    return False



def plot_rewards(filename, window=100):
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            ep, reward = line.strip().split(',')
            rewards.append((int(ep), float(reward)))
    episode_nums, rewards = zip(*rewards)
    plt.plot(episode_nums, rewards, label="Reward")
    rewards_np = np.array(rewards)
    if len(rewards_np) >= window:
        moving_avg = np.convolve(rewards_np, np.ones(window)/window, mode='valid')
        plt.plot(episode_nums[window-1:], moving_avg, label=f"Moving Avg ({window})", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")
    plt.legend()
    plt.show()

def save_line(filename, episode, reward):
    with open(filename, 'a') as f:
        f.write(f"{episode},{reward}\n")


def epsilon_linear(ep_idx):
    return max(HYPERPARAMS["epsilon_end"], HYPERPARAMS["epsilon_start"] - HYPERPARAMS["epsilon_decay"] * ep_idx)



