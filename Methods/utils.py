import numpy as np
import json
import torch

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


def encode_state(state):
    # Example state: {'player': 1, 'banker': 1, 'usable_ace': True}
    return [state["player"], state["banker"], 1 if state["ace"] else 0]


def greedy_policy(Q, state):
    action_values = [Q[(state, a)] for a in [0, 1]]
    if action_values[0] - action_values[1] == 0:
        return np.random.choice([0, 1])
    # 利用: 选择具有最大Q值的动作
    return np.argmax(action_values)


def save_dict_to_json(dic, filename):
    with open("Models/{}.json".format(filename), "w") as f:
        json.dump(dic, f)


def save_net_to_model(net, filename):
    torch.save(net.state_dict(), "Models/{}.pth".format(filename))
