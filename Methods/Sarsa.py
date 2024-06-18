import numpy as np
from collections import defaultdict
from Methods.utils import epsilon_greedy_policy


def sarsa(
    env,
    num_episodes,
    gamma,
    alpha,
    epsilon_start=1.0,
    epsilon_end=0.001,
    epsilon_decay=0.995,
):
    Q = defaultdict(float)
    epsilon = epsilon_start
    for k in range(num_episodes):
        if k % 50 == 0:
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
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
