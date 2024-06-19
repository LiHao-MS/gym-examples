import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from Methods.utils import encode_state
from matplotlib.ticker import MaxNLocator

# 使用同一色彩映射和范围
cmap = "viridis"

def plot_value_function_from_dict(value_dict, title="Value Function"):
    # 玩家可能的点数范围是12到21，庄家展示的牌可能是从1到10
    player_range = np.arange(2, 23).tolist()
    dealer_range = np.arange(1, 11).tolist()
    usable_ace = [False, True]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 11), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    plt.tight_layout()

    # 创建3D图的两个子图，对应有无可用Ace
    for idx, ace in enumerate(usable_ace):
        x, y = np.meshgrid(player_range, dealer_range)
        z = np.array(
            [
                [
                    max(
                        value_dict.get(
                            (
                                frozenset(
                                    {
                                        "player": player,
                                        "dealer": dealer,
                                        "ace": ace,
                                    }.items()
                                ),
                                0,
                            ),
                            0,
                        ),
                        value_dict.get(
                            (
                                frozenset(
                                    {
                                        "player": player,
                                        "dealer": dealer,
                                        "ace": ace,
                                    }.items()
                                ),
                                1,
                            ),
                            0,
                        ),
                    )
                    for player in player_range
                ]
                for dealer in dealer_range
            ]
        )

        ax = axes[idx]
        ax.plot_surface(x, y, z, cmap=cmap)

        ax.set_ylabel("Dealer Showing", fontsize=24)
        ax.set_xlabel("Player Hand", fontsize=24)
        ax.set_zlabel("Value", fontsize=24)
        ax.set_title(f"Usable Ace: {ace}", fontsize=26)
        ax.view_init(ax.elev, -120)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # fig.suptitle(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()


def plot_value_function_from_dqn(model, title="DQN Value Function"):
    player_range = np.arange(2, 23)  # 玩家点数从2到21
    dealer_range = np.arange(1, 11)  # 庄家展示的牌从1到10
    usable_ace = [False, True]  # 是否有可用的Ace

    # 创建3D图的两个子图，对应有无可用Ace
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 11), subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    plt.tight_layout()

    for idx, ace in enumerate(usable_ace):  # 使用enumerate获取索引和是否有可用Ace
        state_values = {}
        for player in player_range:
            for dealer in dealer_range:
                state = {"player": player, "dealer": dealer, "ace": ace}
                state_tensor = torch.tensor(
                    [encode_state(state)], dtype=torch.float32
                )
                with torch.no_grad():
                    value = (
                        model(state_tensor).max(1)[0].item()
                    )  # 获取给定状态的最大Q值
                state_values[(player, dealer)] = value

        x, y = np.meshgrid(player_range, dealer_range)  # 创建网格
        z = np.array(
            [
                [state_values.get((player, dealer), 0) for player in player_range]
                for dealer in dealer_range
            ]
        )

        ax = axes[idx]  # 根据索引选择对应的子图
        ax.plot_surface(x, y, z, cmap=cmap)  # 绘制3D图

        ax.set_ylabel("Dealer Showing", fontsize=24)
        ax.set_xlabel("Player Hand", fontsize=24)
        ax.set_zlabel("Value", fontsize=24)
        ax.set_title(f"Usable Ace: {ace}", fontsize=26)
        ax.view_init(ax.elev, -120)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # fig.suptitle(title)
    plt.savefig("pics/{}.png".format(title))
    # plt.show()
