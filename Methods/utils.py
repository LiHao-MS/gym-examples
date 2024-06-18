import numpy as np
import pickle
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


def save_dict_to_pickle(dic, filename):
    with open("Models/{}.json".format(filename), "w") as f:
        pickle.dump(dic, f)


def save_net_to_model(net, filename):
    torch.save(net.state_dict(), "Models/{}.pth".format(filename))


def load_dict_from_pickle(filename):
    # Load the dictionary from a pickle file
    with open(f"Models/{filename}.pkl", "rb") as f:
        return pickle.load(f)
    
def load_net_from_model(net, filename):
    return net.load_state_dict(torch.load(f"Models/{filename}.pth"))   
