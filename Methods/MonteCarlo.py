import numpy as np
from collections import defaultdict
import gymnasium as gym
from Methods.utils import epsilon_greedy_policy

def run_episode(env, Q, epsilon):
    state, _ = env.reset()
    done = False
    episode = []
    state = frozenset(state.items())
    while not done:
        action = epsilon_greedy_policy(Q, state, epsilon)
        next_state, reward, done, _, info = env.step(action)
        episode.append((state, action, reward))
        next_state = frozenset(next_state.items())
        state = next_state

    return episode


# First Vist
def monte_carlo_prediction(env, num_episodes, gamma):
    Q = defaultdict(float)
    N = defaultdict(int)
    epsilon = 0.3
    for k in range(num_episodes):
        if k % 100 == 0:
            epsilon = max(0.01, epsilon - 0.01)
        episode = run_episode(env, Q, epsilon)
        G = 0
        first_occurrences = {}
        for i, (state, action, reward) in enumerate(episode):
            if (state, action) not in first_occurrences:
                first_occurrences[(state, action)] = i

        # 反向遍历，用首次出现的索引进行G值的计算
        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = reward + gamma * G
            if i == first_occurrences[(state, action)]:  # 检查是否为首次出现
                N[(state, action)] += 1
                Q[(state, action)] += (G - Q[(state, action)]) / N[(state, action)]

    return Q
