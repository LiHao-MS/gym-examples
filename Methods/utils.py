import numpy as np


# ε-greedy策略
def epsilon_greedy_policy(Q, state, epsilon):

    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])  # 探索: 随机选择动作
    else:
        action_values = [Q[(state, a)] for a in [0, 1]]
        if action_values[0] - action_values[1] == 0:
            return np.random.choice([0, 1])
        # 利用: 选择具有最大Q值的动作
        return np.argmax(action_values)
