import numpy as np
from collections import defaultdict
from Methods.utils import epsilon_greedy_policy

def sarsa(env, num_episodes, gamma, alpha, epsilon):
    Q = defaultdict(float)

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = frozenset(state.items())
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done, _, info = env.step(action)
            next_state = frozenset(next_state.items())
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state = next_state
            action = next_action

    return Q
