import Blackjack
import gymnasium as gym
import numpy as np

env = gym.make("Blackjack/Blackjack-v0")


seed = 0
state, _ = env.reset(seed=seed)
done = False
episode = []

while not done:
    action = np.random.choice([0, 1])
    next_state, reward, done, _, info = env.step(action)
    episode.append((state, action, reward))
    state = next_state

