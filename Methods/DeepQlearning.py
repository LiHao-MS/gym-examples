import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


def epsilon_greedy_policy(state, epsilon, policy_net, env):
    if random.random() > epsilon:
        state = torch.tensor([state], dtype=torch.float)
        with torch.no_grad():
            return policy_net(state).max(1)[1].item()
    else:
        return env.action_space.sample()


def optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.tensor(state_batch, dtype=torch.float)
    action_batch = torch.tensor(action_batch, dtype=torch.long)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)
    done_batch = torch.tensor(done_batch, dtype=torch.float)

    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (
        next_state_values * gamma * (1 - done_batch)
    ) + reward_batch

    loss = nn.functional.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_dqn_blackjack(
    env,
    num_episodes=1000,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    lr=0.001,
):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    policy_net = DQN(input_size, output_size)
    target_net = DQN(input_size, output_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = []

    epsilon = epsilon_start
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = epsilon_greedy_policy(state, epsilon, policy_net, env)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma)
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        if episode % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}: Reward {reward}, Epsilon {epsilon}")

    env.close()
    print("Training completed.")