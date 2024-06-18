import Blackjack
import gymnasium as gym
import numpy as np
from Methods import monte_carlo_prediction, sarsa, train_dqn_blackjack, q_learning
from draw import (
    plot_value_function_from_dict,
    plot_value_function_from_dqn,
    plot_training_info,
)

env = gym.make("Blackjack/Blackjack-v0")


# 依次运行四个方法
def main():
    num_episodes = 10000
    gamma = 0.99
    Q1 = monte_carlo_prediction(env, num_episodes=num_episodes, gamma=gamma, seed=0)
    plot_value_function_from_dict(Q1, title="MC Prediction Value Function")
    Q2 = sarsa(env, num_episodes=num_episodes, gamma=gamma, alpha=0.1, epsilon=0.1)
    plot_value_function_from_dict(Q2, title="Sarsa Prediction Value Function")
    Q3 = q_learning(env, num_episodes=num_episodes, gamma=gamma, alpha=0.1, epsilon=0.1)
    plot_value_function_from_dict(Q3, title="Q-Learning Value Function")
    policy_net, rewards, epsilons = train_dqn_blackjack(
        env, num_episodes=num_episodes, gamma=gamma
    )
    plot_value_function_from_dqn(policy_net, env, title="DQN Value Function")
    plot_training_info(rewards, epsilons)

if __name__ == "__main__":
    main()
    env.close()
