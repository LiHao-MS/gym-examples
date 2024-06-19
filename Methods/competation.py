import numpy as np
import torch
from Methods.utils import greedy_policy, encode_state

def compare_methods(env, methods, num_episodes=1000):
    """
    Compare multiple reinforcement learning methods in the given environment.

    Args:
    - env: An environment instance compatible with the methods' action selection.
    - methods: A dictionary of methods with their name and strategy (could be a dictionary or a model).
    - num_episodes: Number of episodes to run for comparison.

    Returns:
    - results: A dictionary with method names as keys and average rewards as values.
    """
    results = {}
    for name, method in methods.items():
        total_rewards = []
        for episode in range(50000):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Select action based on the type of method (dict or model)
                if isinstance(method, dict):
                    state = frozenset(state.items())
                    # Assuming method is a dictionary with state-action value pairs
                    action = np.argmax(
                        [method.get((state, a), 0) for a in [0, 1]]
                    )
                elif isinstance(method, torch.nn.Module):
                    # Assuming method is a neural network model
                    with torch.no_grad():
                        state_tensor = torch.tensor(
                            encode_state(state), dtype=torch.float32
                        ).unsqueeze(0)
                        action_values = method(state_tensor)
                        action = torch.argmax(action_values).item()

                # Take action in the environment
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

        # Compute the average reward
        average_reward = np.mean(total_rewards)
        results[name] = average_reward
        print(
            f"Num_episodes: {num_episodes}, Method: {name}, Average Reward: {average_reward}"
        )

    return results


# Example usage:
# Define your environment and methods (you should have these ready or loaded)
# env = YourEnvironment()
# methods = {
#     "Monte Carlo": mc_policy_dict,
#     "SARSA": sarsa_policy_dict,
#     "DQN": dqn_model  # Assuming this is a trained PyTorch model
# }

# results = compare_methods(env, methods)
