import Blackjack
import gymnasium as gym
import numpy as np
import json
import torch
from Methods import (
    monte_carlo_prediction,
    sarsa,
    train_dqn_blackjack,
    q_learning,
    compare_methods,
    save_dict_to_json,
    save_net_to_model
)
from draw import (
    plot_value_function_from_dict,
    plot_value_function_from_dqn,
    plot_training_info,
    plot_optimal_policy_from_dic,
    plot_optimal_policy_from_dqn,
)


env = gym.make("Blackjack/Blackjack-v0")


# 依次运行四个方法
def main():
    num_episodes_array = [10000, 50000, 100000, 200000]
    gamma = 0.99
    for num_episodes in num_episodes_array:
        Q1 = monte_carlo_prediction(env, num_episodes=num_episodes, gamma=gamma)
        plot_value_function_from_dict(Q1, title="MC Prediction Value Function - {} Episodes".format(num_episodes))
        # plot_optimal_policy_from_dic(Q1, title="MC Prediction Optimal Value Function - {} Episodes".format(num_episodes))
        Q2 = sarsa(env, num_episodes=num_episodes, gamma=gamma, alpha=0.1, epsilon=0.1)
        plot_value_function_from_dict(Q2, title="Sarsa Prediction Value Function - {} Episodes".format(num_episodes))
        # plot_optimal_policy_from_dic(Q2, title="Sarsa Prediction Optimal Value Function - {} Episodes".format(num_episodes))
        Q3 = q_learning(env, num_episodes=num_episodes, gamma=gamma, alpha=0.1, epsilon=0.1)
        plot_value_function_from_dict(Q3, title="Q-Learning Value Function - {} Episodes".format(num_episodes))
        policy_net, rewards, epsilons = train_dqn_blackjack(
            env, num_episodes=num_episodes, gamma=gamma
        )
        plot_value_function_from_dqn(policy_net, title="DQN Value Function - {} Episodes".format(num_episodes))
        # plot_optimal_policy_from_dqn(policy_net, title="DQN Optimal Policy - {} Episodes".format(num_episodes))
        plot_training_info(rewards, epsilons, title="DQN Training Performance - {} Episodes".format(num_episodes))
        methods = {"MC": Q1, "Sarsa": Q2, "Q-Learning": Q3, "DQN": policy_net}
        compare_methods(env, methods, num_episodes=1000)
        
        save_dict_to_json(Q1, "MC_{}_value.json".format(num_episodes))
        save_dict_to_json(Q2, "Sarsa_{}_value.json".format(num_episodes))
        save_dict_to_json(Q3, "Q_Learning_{}_value.json".format(num_episodes))
        save_net_to_model(policy_net, "DQN_{}_model.pth".format(num_episodes))
                          


if __name__ == "__main__":
    main()
    env.close()
    print("Done")
