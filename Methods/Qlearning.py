import numpy as np
from collections import defaultdict
from Methods.utils import epsilon_greedy_policy


def q_learning(env, num_episodes, gamma, alpha, epsilon):
    Q = defaultdict(float)  # 初始化Q值字典，默认值为float类型

    for episode in range(num_episodes):
        state, _ = env.reset()  # 重置环境，获取初始状态
        done = False
        state = frozenset(state.items())
        while not done:
            action = epsilon_greedy_policy(
                Q, state, epsilon
            )  # 根据当前Q值和epsilon-greedy策略选择动作
            next_state, reward, done, _, info = env.step(action)  # 执行动作，获取新状态和奖励
            next_state = frozenset(next_state.items())
            # Q-Learning核心更新公式
            best_next_action = np.argmax(
                [Q[(next_state, a)] for a in [0, 1]]
            )  # 选择下一个状态的最佳动作
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, best_next_action)] - Q[(state, action)]
            )
            state = next_state  # 更新状态

    return Q
