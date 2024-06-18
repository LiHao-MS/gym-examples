import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np


def plot_value_function_from_dict(value_dict, title="Value Function"):
    # 玩家可能的点数范围是12到21，庄家展示的牌可能是从1到10
    player_range = np.arange(12, 22)
    dealer_range = np.arange(1, 11)
    usable_ace = [False, True]

    fig, axes = plt.subplots(nrows=2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()

    # 创建3D图的两个子图，对应有无可用Ace
    for idx, ace in enumerate(usable_ace):
        x, y = np.meshgrid(dealer_range, player_range)
        z = np.array(
            [
                [value_dict.get((player, dealer, ace), 0) for dealer in dealer_range]
                for player in player_range
            ]
        )

        ax = axes[idx]
        ax.plot_surface(x, y, z, cmap="viridis")

        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Hand")
        ax.set_zlabel("Value")
        ax.set_title(f"Usable Ace: {ace}")
        ax.view_init(ax.elev, -120)

    fig.suptitle(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()


def plot_value_function_from_dqn(model, env, title="DQN Value Function"):
    player_range = np.arange(2, 22)  # 玩家点数从2到21
    dealer_range = np.arange(1, 11)  # 庄家展示的牌从1到10
    usable_ace = [False, True]  # 是否有可用的Ace

    # 创建3D图的两个子图，对应有无可用Ace
    fig, axes = plt.subplots(nrows=2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()

    for idx, ace in enumerate(usable_ace):  # 使用enumerate获取索引和是否有可用Ace
        state_values = {}
        for player in player_range:
            for dealer in dealer_range:
                state = (player, dealer, ace)
                state_tensor = torch.tensor(
                    [env.encode_state(state)], dtype=torch.float32
                )
                with torch.no_grad():
                    value = (
                        model(state_tensor).max(1)[0].item()
                    )  # 获取给定状态的最大Q值
                state_values[(player, dealer)] = value

        x, y = np.meshgrid(dealer_range, player_range)  # 创建网格
        z = np.array(
            [
                [state_values.get((player, dealer), 0) for dealer in dealer_range]
                for player in player_range
            ]
        )

        ax = axes[idx]  # 根据索引选择对应的子图
        ax.plot_surface(x, y, z, cmap="viridis")

        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Hand")
        ax.set_zlabel("Value")
        ax.set_title(f"Usable Ace: {ace}")
        ax.view_init(ax.elev, -120)

    fig.suptitle(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()


def plot_training_info(rewards, epsilons, title="Training Performance"):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward", color=color)
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    ax2.set_ylabel("Epsilon", color=color)
    ax2.plot(epsilons, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()

