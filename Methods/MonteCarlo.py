import numpy as np
from collections import defaultdict
import gymnasium as gym


def run_episode(env, Q, seed, epsilon):
    state = env.reset(seed=seed)
    done = False
    episode = []

    while not done:
        action = epsilon_greedy_policy(Q, state, epsilon)
        next_state, reward, done, _, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    return episode

# ε-greedy策略
def epsilon_greedy_policy(Q, state, epsilon):
    action_values = [Q[(state, a)] for a in [0, 1]]
    if action_values[0] - action_values[1] == 0 or np.random.rand() < epsilon:
        return np.random.choice([0, 1])  # 探索: 随机选择动作
    else:
        # 利用: 选择具有最大Q值的动作
        return np.argmax(action_values)


def monte_carlo_prediction(policy, env, num_episodes, gamma):
    V = defaultdict(float)
    N = defaultdict(int)

    for _ in range(num_episodes):
        episode = run_episode(env, policy)
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            Q[state] += (G - Q[state]) / N[state]

    return Q
