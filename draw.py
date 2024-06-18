import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from Methods.utils import encode_state

# 使用同一色彩映射和范围
cmap = "viridis"

def plot_value_function_from_dict(value_dict, title="Value Function"):
    # 玩家可能的点数范围是12到21，庄家展示的牌可能是从1到10
    player_range = np.arange(2, 22).tolist()
    banker_range = np.arange(1, 14).tolist()
    usable_ace = [False, True]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    plt.tight_layout()

    # 创建3D图的两个子图，对应有无可用Ace
    for idx, ace in enumerate(usable_ace):
        x, y = np.meshgrid(banker_range, player_range)
        z = np.array(
            [
                [
                    max(
                        value_dict.get(
                            (frozenset(
                                {"player": player, "banker": banker, "ace": ace}.items()
                            ), 0),
                            0,
                        ),
                        value_dict.get(
                            (frozenset(
                                {"player": player, "banker": banker, "ace": ace}.items()
                            ), 1),
                            0,
                        ),
                    )
                    for banker in banker_range
                ]
                for player in player_range
            ]
        )

        ax = axes[idx]
        ax.plot_surface(x, y, z, cmap=cmap)

        ax.set_xlabel("Banker Showing")
        ax.set_ylabel("Player Hand")
        ax.set_zlabel("Value")
        ax.set_title(f"Usable Ace: {ace}")
        ax.view_init(ax.elev, -120)

    fig.suptitle(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()


def plot_value_function_from_dqn(model, title="DQN Value Function"):
    player_range = np.arange(2, 23)  # 玩家点数从2到21
    banker_range = np.arange(1, 11)  # 庄家展示的牌从1到10
    usable_ace = [False, True]  # 是否有可用的Ace

    # 创建3D图的两个子图，对应有无可用Ace
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 12), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    plt.tight_layout()

    for idx, ace in enumerate(usable_ace):  # 使用enumerate获取索引和是否有可用Ace
        state_values = {}
        for player in player_range:
            for banker in banker_range:
                state = {"player": player, "banker": banker, "ace": ace}
                state_tensor = torch.tensor(
                    [encode_state(state)], dtype=torch.float32
                )
                with torch.no_grad():
                    value = (
                        model(state_tensor).max(1)[0].item()
                    )  # 获取给定状态的最大Q值
                state_values[(player, banker)] = value

        x, y = np.meshgrid(banker_range, player_range)  # 创建网格
        z = np.array(
            [
                [state_values.get((player, banker), 0) for banker in banker_range]
                for player in player_range
            ]
        )

        ax = axes[idx]  # 根据索引选择对应的子图
        ax.plot_surface(x, y, z, cmap=cmap)  # 绘制3D图

        ax.set_xlabel("Banker Showing")
        ax.set_ylabel("Player Hand")
        ax.set_zlabel("Value")
        ax.set_title(f"Usable Ace: {ace}")
        ax.view_init(ax.elev, -120)

    fig.suptitle(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()


def plot_training_info(rewards, epsilons, title="Training Performance"):
    fig, ax1 = plt.subplots()
    plt.tight_layout()

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


def plot_optimal_policy_from_dic(value_dict, title="Optimal Policy Value Function"):
    player_range = np.arange(2, 23)  # 玩家点数从2到21
    banker_range = np.arange(1, 11)  # 庄家展示的牌从1到10

    # 准备数据：遍历玩家和庄家的所有组合，对于每种组合，比较有Ace和无Ace的情况，取最大值
    x, y = np.meshgrid(banker_range, player_range)
    z = np.array(
        [
            [
                max(
                    value_dict.get(
                        (frozenset({"player": player, "banker": banker, "ace": True}.items()), 0), 0
                    ),
                    value_dict.get(
                        (frozenset({"player": player, "banker": banker, "ace": True}.items()), 1), 0
                    ),
                    value_dict.get(
                        (frozenset({"player": player, "banker": banker, "ace": False}.items()), 0), 0
                    ),
                    value_dict.get(
                        (frozenset({"player": player, "banker": banker, "ace": False}.items()), 1), 0
                    ),
                )
                for banker in banker_range
            ]
            for player in player_range
        ]
    )

    # 创建3D图
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10))
    surf = ax.plot_surface(x, y, z, cmap=cmap)
    plt.tight_layout()

    ax.set_xlabel("Banker Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("Maximum Value")
    ax.set_title(title)
    ax.view_init(ax.elev, -120)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 保存图像
    plt.savefig("pics/{}.png".format(title))
    # plt.show()


def plot_optimal_policy_from_dqn(model, title="Optimal Policy Value Function"):
    player_range = np.arange(2, 22)  # 玩家点数从2到21
    banker_range = np.arange(1, 14)  # 庄家展示的牌从1到10

    # 准备数据：遍历玩家和庄家的所有组合，对于每种组合，比较有Ace和无Ace的情况，取最大值
    x, y = np.meshgrid(banker_range, player_range)
    z = np.array(
        [
            [
                max(
                    model(torch.tensor([encode_state({"player": player, "banker": banker, "ace": True})], dtype=torch.float32)).max(1)[0].item(),
                    model(torch.tensor([encode_state({"player": player, "banker": banker, "ace": False})], dtype=torch.float32)).max(1)[0].item()
                )
                for banker in banker_range
            ]
            for player in player_range
        ]
    )

    # 创建3D图
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10))
    surf = ax.plot_surface(x, y, z, cmap=cmap)
    plt.tight_layout()

    ax.set_xlabel("Banker Showing")
    ax.set_ylabel("Player Hand")
    ax.set_zlabel("Maximum Value")
    ax.set_title(title)
    ax.view_init(ax.elev, -120)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 保存图像
    plt.savefig("pics/{}.png".format(title))
    # plt.show(
