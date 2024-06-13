import Blackjack
import gymnasium as gym
import numpy as np
from Methods import monte_carlo_prediction, sarsa, train_dqn, q_learning

env = gym.make("Blackjack/Blackjack-v0")


# 依次运行四个方法
def main():
    Q = monte_carlo_prediction(env, num_episodes=10000, gamma=1.0, seed=0)
    print(Q)
    Q = sarsa(env, num_episodes=10000, gamma=1.0, alpha=0.1, epsilon=0.1)
    print(Q)
    Q = q_learning(env, num_episodes=10000, gamma=1.0, alpha=0.1, epsilon=0.1)
    print(Q)
    train_dqn(env, num_episodes=10000, gamma=1.0, epsilon=0.1)
    
